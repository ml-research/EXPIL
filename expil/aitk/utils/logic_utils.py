# Created by jing at 30.05.23

"""
Root utils file, only import modules that don't belong to this project.
"""
import itertools
import os
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
import config
from expil.aitk.utils import log_utils, file_utils, math_utils
from expil.aitk.utils.fol import bk


def get_index_by_predname(pred_str, atoms):
    indices = []
    for p_i, p_str in enumerate(pred_str):
        p_indices = []
        for i, atom in enumerate(atoms):
            if atom.pred.name == p_str:
                p_indices.append(i)
        indices.append(p_indices)
    return indices


def data_ordering(data):
    data_ordered = torch.zeros(data.shape)
    delta = data[:, :, :3].max(dim=1, keepdims=True)[0] - data[:, :, :3].min(dim=1, keepdims=True)[0]
    order_axis = torch.argmax(delta, dim=2)
    for data_i in range(len(data)):
        data_order_i = data[data_i, :, order_axis[data_i]].sort(dim=0)[1].squeeze(1)
        data_ordered[data_i] = data[data_i, data_order_i, :]

    return data_ordered


def convert_data_to_tensor(args, od_res):
    if os.path.exists(od_res):
        pm_res = torch.load(od_res)
        pos_pred = pm_res['pos_res']
        neg_pred = pm_res['neg_res']
    else:
        raise ValueError
    # data_files = glob.glob(str(pos_dataset_folder / '*.json'))
    # data_tensors = torch.zeros((len(data_files), args.e, 9))
    # for d_i, data_file in enumerate(data_files):
    #     with open(data_file) as f:
    #         data = json.load(f)
    #     data_tensor = torch.zeros(1, args.e, 9)
    #     for o_i, obj in enumerate(data["objects"]):
    #
    #         data_tensor[0, o_i, 0:3] = torch.tensor(obj["position"])
    #         if "blue" in obj["material"]:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([0, 0, 1])
    #         elif "green" in obj["material"]:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([0, 1, 0])
    #         else:
    #             data_tensor[0, o_i, 3:6] = torch.tensor([1, 0, 0])
    #         if "sphere" in obj["material"]:
    #             data_tensor[0, o_i, 6] = 0.99
    #         if "cube" in obj["material"]:
    #             data_tensor[0, o_i, 7] = 0.99
    #         data_tensor[0, o_i, 8] = 0.99
    #     data_tensors[d_i] = data_tensor[0]

    return pos_pred, neg_pred


def vertex_normalization(data):
    return data

    if len(data.shape) != 3:
        raise ValueError

    ax = 0
    min_value = data[:, :, ax:ax + 1].min(axis=1, keepdims=True)[0].repeat(1, data.shape[1], 3)
    max_value = data[:, :, ax:ax + 1].max(axis=1, keepdims=True)[0].repeat(1, data.shape[1], 3)
    data[:, :, :3] = (data[:, :, :3] - min_value) / (max_value - min_value + 1e-10)

    ax = 2
    data[:, :, ax] = data[:, :, ax] - data[:, :, ax].min(axis=1, keepdims=True)[0]
    # for i in range(len(data)):
    #     data_plot = np.zeros(shape=(5, 2))
    #     data_plot[:, 0] = data[i, :5, 0]
    #     data_plot[:, 1] = data[i, :5, 2]
    #     chart_utils.plot_scatter_chart(data_plot, config.buffer_path / "hide", show=True, title=f"{i}")
    return data


def sorted_clauses(clause_with_scores, args):
    if len(clause_with_scores) > 0:
        c_sorted = sorted(clause_with_scores, key=lambda x: x[1][2], reverse=True)
        # for c in c_sorted:
        #     log_utils.add_lines(f"clause: {c[0]} {c[1]}", args.log_file)
        return c_sorted
    else:
        return []


def extract_clauses_from_max_clause(bs_clauses, args):
    clauses = []
    if len(bs_clauses) == 0:
        return clauses

    for bs_clause in bs_clauses:
        clauses.append(bs_clause[0])
        log_utils.add_lines(f"add max clause: {bs_clause[0]}", args.log_file)
    return clauses


def top_select(bs_clauses, args):
    # all_c = bs_clauses['sn'] + bs_clauses['nc'] + bs_clauses['sc'] + bs_clauses['nc_good'] + bs_clauses['sc_good'] + \
    #         bs_clauses['uc'] + bs_clauses['uc_good']

    top_clauses = sorted_clauses(bs_clauses, args)
    top_clauses = top_clauses[:args.c_top]
    top_clauses = extract_clauses_from_max_clause(top_clauses, args)
    return top_clauses


def extract_clauses_from_bs_clauses(bs_clauses, c_type, args):
    clauses = []
    if len(bs_clauses) == 0:
        return clauses

    for bs_clause in bs_clauses:
        clauses.append(bs_clause[0])
        if args.show_process:
            if len(bs_clause) == 3:
                positive_score = bs_clause[2][:, config.score_example_index["pos"]]
                negative_score = 1 - bs_clause[2][:, config.score_example_index["neg"]]
                failed_pos_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
                failed_neg_index = ((negative_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
                log_utils.add_lines(
                    f"({c_type}): {bs_clause[0]} suff: {bs_clause[1].reshape(-1)} "
                    f"Failed Pos States: ({len(failed_pos_index)}/{bs_clause[2].shape[0]}) "
                    f"Failed Neg States: ({len(failed_neg_index)}/{bs_clause[2].shape[0]}) ",
                    args.log_file)
    return clauses


def get_pred_names_from_clauses(clause, exclude_objects=False):
    preds = []
    for atom in clause.body:
        pred = atom.pred.name
        if "in" == pred:
            continue
        if exclude_objects:
            terms = [t.name for t in atom.terms if "O" not in t.name]
        else:
            terms = [t.name for t in atom.terms]
        if pred not in preds:
            preds.append([pred, terms])
    return preds


def get_semantic_from_c(clause):
    semantic = []
    semantic += get_pred_names_from_clauses(clause)
    return semantic


def get_independent_clusters(args, lang, clauses):
    if args.show_process:
        print(f"- searching for independent clauses from {len(clauses)} clauses...")

    clauses_with_score = []
    clause_objs = []
    for clause_i, [clause, four_scores, c_scores] in enumerate(clauses):
        clause_objs.append([atom.terms[0].name for atom in clause.body] + [atom.pred.name for atom in clause.body])
    grouped_names_indices = defaultdict(list)
    for i, name_list in enumerate(clause_objs):
        grouped_names_indices[tuple(name_list)].append(i)
    group_indices = list(grouped_names_indices.values())

    clause_clusters = []
    for group_i in range(len(group_indices)):
        clause_clusters.append([[i, clauses[i][0], clauses[i][2]] for i in group_indices[group_i]])
    return clause_clusters


def check_trivial_clusters(clause_clusters):
    clause_clusters_untrivial = []
    for c_clu in clause_clusters:
        is_trivial = False
        if len(c_clu) > 1:
            for c_i, c in enumerate(c_clu):
                clause = c[1]
                clause.body = sorted(clause.body)
                if c_i > 0:
                    if has_same_preds_and_atoms(clause, c_clu[0][1]):
                        is_trivial = True
                        break
                    if not has_same_preds(clause, c_clu[0][1]):
                        is_trivial = True
                        break
        if not is_trivial:
            clause_clusters_untrivial.append(c_clu)
    return clause_clusters_untrivial


def has_same_preds_and_atoms(c1, c2):
    if len(c1.body) != len(c2.body):
        return False
    same_preds = True
    for i in range(len(c1.body)):
        if not same_preds:
            break
        if not c1.body[i].pred.name == c2.body[i].pred.name:
            same_preds = False
        else:
            for j, term in enumerate(c1.body[i].terms):
                if "O" not in term.name:
                    if not term.name == c2.body[i].terms[j].name:
                        same_preds = False
    if same_preds:
        return True
    else:
        return False


def has_same_preds(c1, c2):
    if len(c1.body) != len(c2.body):
        return False
    same_preds = True
    for i in range(len(c1.body)):
        if not c1.body[i].pred.name == c2.body[i].pred.name:
            same_preds = False
    if same_preds:
        return True
    else:
        return False


def sub_lists(l, min_len=0, max_len=None):
    # initializing empty list
    comb = []

    # Iterating till length of list
    if max_len is None:
        max_len = len(l) + 1
    for i in range(min_len, max_len):
        # Generating sub list
        comb += [list(j) for j in itertools.combinations(l, i)]
    # Returning list
    return comb


def update_args(args, data):
    args.train_data = []
    args.test_data = []

    for a_i in range(len(data)):
        data_size = min(args.top_data, len(data[a_i]["pos_data"]), len(data[a_i]["neg_data"]))
        random_pos_indices = torch.randperm(len(data[a_i]["pos_data"]))[:data_size]
        random_neg_indices = torch.randperm(len(data[a_i]["neg_data"]))[:data_size]
        pos_data = data[a_i]["pos_data"][random_pos_indices]
        neg_data = data[a_i]["neg_data"][random_neg_indices]
        train_pos = pos_data
        train_neg = neg_data
        test_pos = pos_data
        test_neg = neg_data

        train_pos = math_utils.closest_one_percent(train_pos, 0.01)
        train_neg = math_utils.closest_one_percent(train_neg, 0.01)
        test_pos = math_utils.closest_one_percent(test_pos, 0.01)
        test_neg = math_utils.closest_one_percent(test_neg, 0.01)
        args.train_data.append([train_pos, train_neg])
        args.test_data.append([test_pos, test_neg])

    args.lark_path = config.root / "expil" / "lark" / "exp.lark"

    args.invented_pred_num = 0
    args.invented_consts_number = 0

    args.last_refs = []
    args.found_ns = False
    bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
    neural_preds = file_utils.load_neural_preds(bk_preds, "bk_pred")
    args.neural_preds = [neural_pred for neural_pred in neural_preds]
    args.p_inv_counter = 0


def update_eval_args(args, pm_prediction_dict, obj_groups, obj_avail):
    args.train_pos = pm_prediction_dict["train_pos"]  # .to(args.device)
    args.train_neg = pm_prediction_dict["train_neg"]  # .to(args.device)
    args.train_group_pos = obj_groups['group_train_pos']
    args.train_group_neg = obj_groups['group_train_neg']
    args.obj_avail_train_pos = obj_avail['obj_avail_train_pos']
    args.obj_avail_train_neg = obj_avail['obj_avail_train_neg']

    args.data_size = len(args.train_pos)
    args.invented_pred_num = 0
    args.last_refs = []
    args.found_ns = False
    args.d = len(config.group_tensor_index)

    # clause generation and predicate invention
    lang_data_path = args.lang_base_path / args.dataset_type / args.dataset

    pi_type = config.pi_type['bk']
    bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
    neural_preds = file_utils.load_neural_preds(bk_preds, pi_type)

    args.neural_preds = [neural_pred for neural_pred in neural_preds]
    args.p_inv_counter = 0
