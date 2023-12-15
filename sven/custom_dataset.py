import os
import abc
import json
import torch
import random
from torch.utils.data import Dataset

from constant import REPO_BINARY_LABELS, IN_REPO_LABEL, OUT_REPO_LABEL
from utils import get_indent

class DatasetBase(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset = list()

        # i'm not sure if we need the vul type here
        # since we dont differentiate between the types of the functions
        # maybe instead of vul type, we should have code type, like "function" 
        if self.args.vul_type is not None:
            vul_types = [self.args.vul_type]

        # vul types in the constant file is just stuff like cwe-119 
        # which I think will point to the data directory
        for i, vul_type in enumerate(vul_types):
            # yeah read the file containing the json
            # So it will join custom_data_train_val + train + functions.jsonl
            with open(os.path.join(args.data_dir, mode, f'{vul_type}.jsonl')) as f:
                lines = f.readlines()
            
            for l, line in enumerate(lines):
                # read in the json line describine the code and its changes
                print(l)
                diff_j = json.loads(line)

                # determine what language we are using
                lang = 'py'

                # these are "sec" and "vul" respectively
                labels = [IN_REPO_LABEL, OUT_REPO_LABEL]

                # retrieve the function sources
                srcs = [diff_j['func_src_after'], diff_j['func_src_before']]

                # when creating the dataset we will choose what kind of diff 
                # level we want in order to generate a mask?
                # default level is mix
                if self.args.diff_level == 'prog':
                    diffs = [None, None]
                elif self.args.diff_level == 'line':
                    diffs = [diff_j['line_changes']['added'], diff_j['line_changes']['deleted']]
                elif self.args.diff_level == 'char':
                    diffs = [diff_j['char_changes']['added'], diff_j['char_changes']['deleted']]
                elif self.args.diff_level == 'mix':
                    diffs = [diff_j['char_changes']['added'], diff_j['line_changes']['deleted']]
                else:
                    raise NotImplementedError()

                # not sure how the zip behavior will work here
                for label, src, changes in zip(labels, srcs, diffs):
                    self.add_data(label, src, changes, i, lang)

    @abc.abstractclassmethod
    def add_data(self, label, src, changes, vul_id):
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return tuple(torch.tensor(t) for t in self.dataset[item])


class PrefixDataset(DatasetBase):
    def __init__(self, args, tokenizer, mode):
        super().__init__(args, tokenizer, mode)

    # create a tensor of the src code, its vulnerability ID(what kind of vulnerability it is),
    # its label ID (0 for vul, 1 for sec) and then its changes. Then append that tensor to our data
    def add_data(self, label, src, changes, vul_id, lang):
        control_id = REPO_BINARY_LABELS.index(label)    
        data = self.get_tensor(src, vul_id, control_id, changes)
        if data is not None:
            self.dataset.append(data)

    # create a tensor using our code, vul_id, control_id, and its changes
    def get_tensor(self, src, vul_id, control_id, changes):
        # tokenize the src code
        be = self.tokenizer.encode_plus(src)
        tokens = be.data['input_ids']

        if len(tokens) > self.args.max_num_tokens: return None
        
        # this will likely always be 1 in our case
        min_changed_tokens = 1

        # i think this is generating the mask for the changes 
        # over the tokens over the source code
        if changes is None:
            weights = [1] * len(tokens)
        else:
            weights = [0] * len(tokens)
            for change in changes:
                char_start = change['char_start']
                char_start_idx = be.char_to_token(char_start)
                char_end = change['char_end']
                char_end_idx = be.char_to_token(char_end-1)
                for char_idx in range(char_start_idx, char_end_idx+1):
                    weights[char_idx] = 1
            if sum(weights) < min_changed_tokens: return None
            if len(tokens) - sum(weights) < min_changed_tokens: return None

        return tokens, weights, control_id, vul_id


# # i don't know if we're going to need this yet
# # depends on this handles the text prompts for when we want to wvaluate our dataset
# class TextPromptDataset(DatasetBase):
#     def __init__(self, args, tokenizer, mode):
#         super().__init__(args, tokenizer, mode)

#     def add_data(self, label, src, changes, vul_id, lang):
#         control_id = BINARY_LABELS.index(label)    
#         if lang == 'py':
#             control = get_indent(src) + '# ' + PROMPTS[control_id]
#         else:
#             control = get_indent(src) + '// ' + PROMPTS[control_id]
#         src = control + src
#         data = self.get_tensor(src, control, changes)
#         if data is not None:
#             self.dataset.append(data)

#     def get_tensor(self, src, control, changes):
#         be = self.tokenizer.encode_plus(src)
#         tokens = be.data['input_ids']

#         if changes is None:
#             labels = tokens[:]
#         else:
#             labels = [-100] * len(tokens)
#             label_set = False
#             for change in changes:
#                 char_start = change['char_start'] + len(control)
#                 char_start_idx = be.char_to_token(char_start)
#                 char_end = change['char_end'] + len(control)
#                 char_end_idx = be.char_to_token(char_end-1)
#                 for i in range(char_start_idx, char_end_idx+1):
#                     labels[i] = tokens[i]
#                     label_set = True
#             if not label_set: return None

#         if len(tokens) > self.args.max_num_tokens: return None
#         return tokens, labels