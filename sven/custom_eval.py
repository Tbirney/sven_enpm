import os
import csv
import json
import torch
import shutil
import argparse
import subprocess
import libcst as cst
from libcst.metadata import PositionProvider
from libcst._position import CodePosition
from collections import OrderedDict

from sven.evaler import LMEvaler, PrefixEvaler, TextPromptEvaler
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import BINARY_LABELS, MODEL_DIRS, CWES_DICT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained_subset', 'prompts', 'gen_1', 'gen_2'], default='trained')
    parser.add_argument('--vul_type', type=str, default='functions')
    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix'], default='prefix')
    parser.add_argument('--model_dir', type=str, default=None)

    parser.add_argument('--data_dir', type=str, default='../custom_data_eval')
    parser.add_argument('--output_dir', type=str, default='../experiments/repo_eval')

    parser.add_argument('--num_gen', type=int, default=25)
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--max_gen_len', type=int, default=300)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.model_type == 'lm':
        if args.model_dir is None:
            args.model_dir = '2b'
        if args.model_dir in MODEL_DIRS:
            args.model_dir = MODEL_DIRS[args.model_dir]

    args.output_dir = os.path.join(args.output_dir, args.output_name, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)

    return args

def get_evaler(args):
    if args.model_type == 'lm':
        evaler = LMEvaler(args)
        controls = ['orig']
    elif args.model_type == 'prefix':
        evaler = PrefixEvaler(args)
        controls = BINARY_LABELS
    else:
        raise NotImplementedError()

    return evaler, controls


def eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario):
    s_out_dir = os.path.join(output_dir, scenario)
    os.makedirs(s_out_dir)
    s_in_dir = os.path.join(data_dir, scenario)
    with open(os.path.join(s_in_dir, 'info.json')) as f:
        info = json.load(f)
    with open(os.path.join(s_in_dir, 'file_context.'+info['language'])) as f:
        file_context = f.read()
    with open(os.path.join(s_in_dir, 'func_context.'+info['language'])) as f:
        func_context = f.read()

    for control_id, control in enumerate(controls):
        set_seed(args)
        with torch.no_grad():
            outputs, output_ids, dup_srcs, non_parsed_srcs = evaler.sample(file_context, func_context, control_id, info['language'])

        out_src_dir = os.path.join(s_out_dir, f'{control}_output')
        os.makedirs(out_src_dir)
        output_ids_j = OrderedDict()
        all_fnames = set()
        for i, (output, output_id) in enumerate(zip(outputs, output_ids)):
            fname = f'{str(i).zfill(2)}.'+info['language']
            all_fnames.add(fname)
            with open(os.path.join(out_src_dir, fname), 'w') as f:
                f.write(output)
            output_ids_j[fname] = output_id
        with open(os.path.join(s_out_dir, f'{control}_output_ids.json'), 'w') as f:
            json.dump(output_ids_j, f, indent=2)
        if info['language'] == 'c':
            shutil.copy2('Makefile', out_src_dir)

        for srcs, name in [(dup_srcs, 'dup'), (non_parsed_srcs, 'non_parsed')]:
            src_dir = os.path.join(s_out_dir, f'{control}_{name}')
            os.makedirs(src_dir)
            for i, src in enumerate(srcs):
                fname = f'{str(i).zfill(2)}.'+info['language']
                with open(os.path.join(src_dir, fname), 'w') as f:
                    f.write(src)

        vuls = set()
        if len(outputs) != 0:
            csv_path = os.path.join(s_out_dir, f'{control}_codeql.csv')
            db_path = os.path.join(s_out_dir, f'{control}_codeql_db')
            codeql_create_db(info, out_src_dir, db_path)
            codeql_analyze(info, db_path, csv_path)
            if vul_type == 'cwe-078':
                filter_cwe78_fps(s_out_dir, control)
            with open(csv_path) as csv_f:
                reader = csv.reader(csv_f)
                for row in reader:
                    if len(row) < 5: continue
                    out_src_fname = row[-5].replace('/', '')
                    vuls.add(out_src_fname)
        secs = all_fnames - vuls

        d = OrderedDict()
        d['vul_type'] = vul_type
        d['scenario'] = scenario
        d['control'] = control
        d['total'] = len(all_fnames)
        d['sec'] = len(secs)
        d['vul'] = len(vuls)
        d['dup'] = len(dup_srcs)
        d['non_parsed'] = len(non_parsed_srcs)
        d['model_type'] = args.model_type
        d['model_dir'] = args.model_dir
        d['temp'] = args.temp

        yield d

def eval_vul(args, evaler, controls, vul_types):
    for vul_type in vul_types:
        data_dir = os.path.join(args.data_dir, vul_type)
        output_dir = os.path.join(args.output_dir, vul_type)
        os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'result.jsonl'), 'w') as f:
            for scenario in list(sorted(os.listdir(data_dir))):
                for d in eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario):
                    s = json.dumps(d)
                    args.logger.info(s)
                    f.write(s+'\n')

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_devices(args)
    set_seed(args)
    args.logger.info(f'args: {args}')

    evaler, controls = get_evaler(args)
    assert args.eval_type in CWES_DICT
    if args.vul_type is not None:
        vul_types = [args.vul_type]
    else:
        vul_types = CWES_DICT[args.eval_type]

    eval_vul(args, evaler, controls, vul_types)

if __name__ == '__main__':
    main()