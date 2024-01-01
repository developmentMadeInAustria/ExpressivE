
import pandas as pd

import torch

from pykeen.regularizers import Regularizer
from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory
from pykeen.trackers import ResultTracker, tracker_resolver

from class_resolver import HintOrType, OptionalKwargs

from typing import Optional, Mapping, Any
import re

from torch import Tensor

from Utils import preprocess_relations


class ExpressivERegularizer(Regularizer):

    __alpha: float
    __batch_size: int
    __apply_rule_confidence: bool
    __tanh_map: bool
    __min_denom: float

    __tracked_rules: [int]
    __iteration: int = 0
    __result_tracker: ResultTracker

    __device: torch.device

    __factory: TriplesFactory
    __rules: pd.DataFrame

    def __init__(
            self,
            dataset: str,
            dataset_kwargs: Optional[Mapping[str, Any]],
            rules: str,
            rules_max_body_atoms: int = 2,
            rule_min_confidence: float = 0.1,
            alpha: float = 1,
            batch_size: int = None,
            apply_rule_confidence = False,
            tanh_map: bool = True,
            min_denom: float = 0.5,
            tracked_rules=None,
            result_tracker: HintOrType[ResultTracker] = None,
            result_tracker_kwargs: OptionalKwargs = None,
            **kwargs
    ) -> None:
        if tracked_rules is None:
            tracked_rules = list()
        kwargs['apply_only_once'] = True
        super().__init__(**kwargs)

        if rules_max_body_atoms > 2:
            raise ValueError("Error: regularizer only for up to two body atoms implemented!")

        if rule_min_confidence > 1:
            raise ValueError("Error: minimum rule confidence can't be greater than one!")

        if alpha < 0:
            raise ValueError("Error: alpha must be greater than zero!")
        self.__alpha = alpha

        self.__batch_size = batch_size
        self.__apply_rule_confidence = apply_rule_confidence
        self.__tanh_map = tanh_map
        self.__min_denom = min_denom

        self.__tracked_rules = tracked_rules
        if self.__tracked_rules is None:
            self.__tracked_rules = []
        self.__result_tracker = tracker_resolver.make(query=result_tracker, pos_kwargs=result_tracker_kwargs)

        if torch.cuda.is_available():
            # pykeen is optimized for single gpu usage
            # after setting CUDA_VISIBLE_DEVICES to e.g. 1, gpu 1 can only be accessed via cuda:0
            self.__device = torch.device('cuda:0')
        else:
            self.__device = torch.device('cpu')

        # Future Improvement: Move loading to a separate class to allow loading of different formats (AnyBURL, AMIE)

        # get dataset and triples factory
        mutable_dataset_kwargs = dict(dataset_kwargs)
        mutable_dataset_kwargs['eager'] = True
        dataset = get_dataset(dataset=dataset, dataset_kwargs=mutable_dataset_kwargs)
        # noinspection PyTypeChecker
        self.__factory: TriplesFactory = dataset.training

        # read rules and format
        rule_df = pd.read_csv(rules, sep='\t', names=['predictions', 'support', 'confidence', 'rule'])
        rule_df[['head', 'body']] = rule_df['rule'].str.split(' <= ', expand=True)
        rule_df['body'] = rule_df['body'].str.split(', ', expand=False)
        rule_df['body'] = rule_df['body'].apply(lambda x: x if x[0] != '' else [])
        rule_df['body_count'] = rule_df['body'].str.len()
        rule_df.drop('rule', axis=1)

        # filter
        rule_df = rule_df[rule_df['confidence'] >= rule_min_confidence] # only min confidence rules
        rule_df = rule_df[rule_df['body_count'] <= rules_max_body_atoms] # max body atoms
        rule_df = rule_df[rule_df['body'].apply(self.__no_const_body)] # no constants in body
        rule_df = rule_df[rule_df['head'].apply(self.__no_const_head)]  # no constants in head

        # add ids
        rule_df['body_ids'] = rule_df['body'].apply(self.__body_ids)
        rule_df['head_id'] = rule_df['head'].apply(self.__head_id)
        rule_df['ids'] = rule_df.apply(lambda x: set(x.body_ids).union([x.head_id]), axis=1)

        self.__rules = rule_df

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # TODO: If lots of rules, split dataframe and parallelize
        self.__iteration += 1 # TODO: Sync with epochs

        if self.__batch_size is None:
            rules = self.__rules
        else:
            rules = self.__rules.sample(self.__batch_size)

        rules_loss = None
        for idx, row in rules.iterrows():
            # use iterrows() as apply() + sum() throws error:
            # "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."

            rule_multiplier = row['confidence'] if self.__apply_rule_confidence else 1.0
            rule_loss = rule_multiplier * self.__compute_loss(row, x)

            if rules_loss is None:
                rules_loss = rule_loss
            else:
                rules_loss += rule_loss

            if idx in self.__tracked_rules:
                self.__result_tracker.log_metrics({"rule_{}_loss".format(idx): rule_loss}, step=self.__iteration)

        self.__result_tracker.log_metrics({"rules_loss": rules_loss}, step=self.__iteration)
        return self.__alpha * rules_loss

    def __no_const_body(self, atoms: [str]) -> bool:
        arguments = map(self.__extract_arguments, atoms)
        no_const_argument = map(self.__no_const_arguments, arguments)

        return all(no_const_argument)

    def __no_const_head(self, atom: str) -> bool:
        arguments = self.__extract_arguments(atom)
        return self.__no_const_arguments(arguments)

    def __extract_arguments(self, atom: str) -> str:
        pattern = re.compile('\([^)]*\)')
        regex_result = pattern.search(atom)
        arguments = regex_result.group(0)

        return arguments

    def __no_const_arguments(self, arguments: str) -> bool:
        pattern = re.compile('\([A-Z],[A-Z]\)')
        regex_result = pattern.match(arguments)
        return regex_result is not None

    def __body_ids(self, atoms: [str]) -> [int]:
        relations = list(map(self.__extract_relation, atoms))
        return self.__factory.relations_to_ids(relations)

    def __head_id(self, atom) -> int:
        relation = self.__extract_relation(atom)
        ids = list(self.__factory.relations_to_ids([relation]))
        return ids[0]

    def __extract_relation(self, atom: str) -> str:
        pattern = re.compile('[^(]*')
        regex_result = pattern.search(atom)
        relation = regex_result.group(0)

        return relation

    def __compute_loss(self, rule, weights) -> torch.FloatTensor:
        body_args = map(self.__extract_arguments, rule['body'])
        body_args = list(map(lambda args: args[1:-1], body_args))
        head_args = self.__extract_arguments(rule['head'])
        head_args = head_args[1:-1]

        # TODO: Implement body_count == 0

        if rule['body_count'] == 1:
            return self.__compute_loss_one_atom(body_args, head_args, rule['body_ids'], rule['head_id'], weights)
        elif rule['body_count'] == 2:
            return self.__compute_loss_two_atoms(body_args, head_args, rule['body_ids'], rule['head_id'], weights)

        # TODO: Either raise error/ put assertion failure as this can't happen
        return 0

    # TODO: Move to separate class
    def __compute_loss_one_atom(self, body_args, head_args, body_ids, head_id, weights) -> torch.FloatTensor:
        if head_args != 'X,Y':
            print("Only rules with head (X,Y) supported!")
            return torch.FloatTensor([0])

        if body_args[0] == 'X,Y':
            # hierarchy: r(x,y) -> s(x,y) = r(x,y) and i(y,y) -> s(x,y)
            body_weights = weights[body_ids[0], :]
            embedding_dim = int(len(body_weights) / 6)
            self_loop = torch.cat((torch.zeros(embedding_dim*4, device=self.__device),
                                   torch.ones(embedding_dim*2, device=self.__device)))
            head_weights = weights[head_id, :]
            rule_weights = torch.stack((body_weights, self_loop, head_weights))
            return self.general_composition_loss(rule_weights)
        else:
            # inversion: r(x,y) -> r(y,x) = r(x,y) and i(y,y) -> r(y,x)
            print("Loss for inversion rules not implemented yet")
            return 0

    def __compute_loss_two_atoms(self, body_args, head_args, body_ids, head_id, weights) -> torch.FloatTensor:
        if head_args != 'X,Y':
            print("Only rules with head (X,Y) supported!")
            return torch.FloatTensor([0])

        chain_order = self.__compute_chain_order(body_args, [], 'X')

        body_indices = torch.tensor(body_ids, device=self.__device)
        body_weights = torch.index_select(weights, 0, body_indices)
        flipped_weights = self.__flip_weights(body_weights)
        chain_order_access = [[chain_order[0]] * body_weights.size()[1], [chain_order[1]] * body_weights.size()[1]]
        loss_body_weights = torch.where(torch.tensor(chain_order_access, device=self.__device), body_weights, flipped_weights)

        head_weights = weights[head_id, :]
        head_weights = torch.reshape(head_weights, (1, -1))

        rule_weights = torch.cat((loss_body_weights, head_weights), dim=0)
        return self.general_composition_loss(rule_weights)

    def __compute_chain_order(self, body_args, current_chain, prev_dangling_atom) -> [bool]:
        """
        Determines for which relations the order of the variables is reversed - e.g. A,X instead of X,A .
        Applied recursively.

        :param body_args: List of arguments of the different body arguments - e.g. ["X,A","B,A","B,Y"]
        :param current_chain: List of current order in previous recursion.
        :param prev_dangling_atom: Dangling atom in previous relation (atom). (Recursion!)
        :return: A list of which relations have the order of variables reversed.
        """

        if len(body_args) == 1:
            ordered = body_args[0][0] == prev_dangling_atom
            return current_chain + [ordered]
        else:
            if body_args[0][0] == prev_dangling_atom:
                # New variable on right side
                return self.__compute_chain_order(body_args[1:], current_chain + [True], body_args[0][2])
            else:
                # New variable on left side
                return self.__compute_chain_order(body_args[1:], current_chain + [False], body_args[0][2])

    def __flip_weights(self, weights: torch.Tensor) -> torch.Tensor:
        num_weights = weights.size()[0]
        embedding_dim = int(weights.size()[1] / 6)

        idx_list = []
        counter = 0
        for i in range(0, num_weights * 3):
            idx_list += list(range(embedding_dim + counter * (2 * embedding_dim), (counter + 1) * (2 * embedding_dim)))
            idx_list += list(range(counter * (2 * embedding_dim), embedding_dim + counter * (2 * embedding_dim)))

            counter += 1

        single_dim_weights = torch.flatten(weights)
        single_dim_flipped = single_dim_weights[idx_list]
        flipped = single_dim_flipped.view(num_weights, -1)
        return flipped

    def general_composition_loss(self, weights, self_loop=False) -> torch.FloatTensor:
        """
        Computes the general composition loss (for two body atoms)

        :param self_loop: Determines, if the second weight (second body atom) is the self loop relation.
        :param weights: Weights for all three relations (two body relations, one head relation)
        :return: The loss for general composition
        """

        rel_1 = preprocess_relations(weights[0, :], tanh_map=self.__tanh_map, min_denom=self.__min_denom)
        rel_2 = weights[1, :]
        if not self_loop:
            rel_2 = preprocess_relations(rel_2, tanh_map=self.__tanh_map, min_denom=self.__min_denom)
        rel_3 = preprocess_relations(weights[2, :], tanh_map=self.__tanh_map, min_denom=self.__min_denom)
        d3_h, d3_t, c3_h, c3_t, s3_h, s3_t = rel_3

        # Calculate four corner points of head relation
        corner1_x = ((c3_t-c3_h) + (d3_t-d3_h)) / (s3_t-s3_h) # top right
        corner1_y = c3_t + d3_t + s3_h * corner1_x

        corner2_x = ((c3_t-c3_h) + (-d3_t-d3_h)) / (s3_t-s3_h) # bottom right
        corner2_y = c3_t - d3_t + s3_h * corner2_x

        corner3_x = ((c3_t - c3_h) + (-d3_t + d3_h)) / (s3_t - s3_h)  # bottom left
        corner3_y = c3_t - d3_t + s3_h * corner3_x

        corner4_x = ((c3_t - c3_h) + (d3_t + d3_h)) / (s3_t - s3_h)  # top left
        corner4_y = c3_t + d3_t + s3_h * corner4_x

        # Check each inequality with each corner point
        corner1_loss = self.general_composition_corner_loss(corner1_x, corner1_y, rel_1, rel_2)
        corner2_loss = self.general_composition_corner_loss(corner2_x, corner2_y, rel_1, rel_2)
        corner3_loss = self.general_composition_corner_loss(corner3_x, corner3_y, rel_1, rel_2)
        corner4_loss = self.general_composition_corner_loss(corner4_x, corner4_y, rel_1, rel_2)

        return corner1_loss + corner2_loss + corner3_loss + corner4_loss

    def general_composition_corner_loss(self, corner_x, corner_y, rel_1, rel_2) -> Tensor:
        d1_h, d1_t, c1_h, c1_t, s1_h, s1_t = rel_1
        d2_h, d2_t, c2_h, c2_t, s2_h, s2_t = rel_2

        zero_loss = torch.zeros(corner_x.size(), device=self.__device)
        ones = torch.ones(corner_x.size(), device=self.__device)

        # TODO: Discuss aggregation function.
        eq1_loss = abs(corner_x - corner_y*s1_t*s2_t - c2_h*s1_t - c1_h) - d2_h*s1_t - d1_h
        eq1_loss = torch.mean(torch.maximum(zero_loss, eq1_loss))

        eq2_loss = abs(corner_y*s2_t + c2_h - corner_x*s1_h - c1_t) - d1_t - d2_h
        eq2_loss = torch.mean(torch.maximum(zero_loss, eq2_loss))

        eq3_loss = abs(corner_y - corner_x*s1_h*s2_h - c1_t*s2_h - c2_t) - d1_t*s2_h - d2_t
        eq3_loss = torch.mean(torch.maximum(zero_loss, eq3_loss))

        eq4_loss = abs(corner_y + (c1_h - corner_x)*s2_h/s1_t - c2_t) - d1_h*s2_h/s1_t - d2_t
        eq4_loss = torch.mean(torch.maximum(zero_loss, eq4_loss))

        eq5_loss = abs(corner_x*(ones - s1_h*s1_t) - c1_t*s1_t - c1_h) - d1_t*s1_t - d1_h
        eq5_loss = torch.mean(torch.maximum(zero_loss, eq5_loss))

        eq6_loss = abs(corner_y*(ones - s2_h*s2_t) - c2_h*s2_h - c2_t) - d2_h*s2_h - d2_t
        eq6_loss = torch.mean(torch.maximum(zero_loss, eq6_loss))

        max_loss = max(eq1_loss, eq2_loss, eq3_loss, eq4_loss, eq5_loss, eq6_loss)
        return max_loss


if __name__ == '__main__':
    reg = ExpressivERegularizer("WN18RR", dataset_kwargs={}, rules="../Rules/WN18RR-Max5-Run1/WN18RR-1000", rules_max_body_atoms=3, rule_min_confidence=0.1)
    reg.forward(torch.FloatTensor([[0.1, 0.2, 0.1, 0.3, 0.1, 0.4], [0.1, 0.5, 0.1, 0.6, 0.1, 0.7], [0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
                                   [0.1, 0.2, 0.1, 0.3, 0.1, 0.4], [0.1, 0.5, 0.1, 0.6, 0.1, 0.7], [0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
                                   [0.1, 0.2, 0.1, 0.3, 0.1, 0.4], [0.1, 0.5, 0.1, 0.6, 0.1, 0.7], [0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
                                   [0.1, 0.2, 0.1, 0.3, 0.1, 0.4], [0.1, 0.5, 0.1, 0.6, 0.1, 0.7], [0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
                                   [0.1, 0.2, 0.1, 0.3, 0.1, 0.4], [0.1, 0.5, 0.1, 0.6, 0.1, 0.7], [0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
                                   [0.1, 0.2, 0.1, 0.3, 0.1, 0.4], [0.1, 0.5, 0.1, 0.6, 0.1, 0.7], [0.1, 0.8, 0.1, 0.9, 0.1, 0.10],]))

    reg.forward(torch.FloatTensor(
        [[0.1, 0.2, 0.1, 0.3, 0.1, 0.4, 0.1, 0.2, 0.1, 0.3, 0.1, 0.4],
         [0.1, 0.5, 0.1, 0.6, 0.1, 0.7, 0.1, 0.5, 0.1, 0.6, 0.1, 0.7],
         [0.1, 0.8, 0.1, 0.9, 0.1, 0.10, 0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
         [0.1, 0.2, 0.1, 0.3, 0.1, 0.4, 0.1, 0.2, 0.1, 0.3, 0.1, 0.4],
         [0.1, 0.5, 0.1, 0.6, 0.1, 0.7, 0.1, 0.5, 0.1, 0.6, 0.1, 0.7],
         [0.1, 0.8, 0.1, 0.9, 0.1, 0.10, 0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
         [0.1, 0.2, 0.1, 0.3, 0.1, 0.4, 0.1, 0.2, 0.1, 0.3, 0.1, 0.4],
         [0.1, 0.5, 0.1, 0.6, 0.1, 0.7, 0.1, 0.5, 0.1, 0.6, 0.1, 0.7],
         [0.1, 0.8, 0.1, 0.9, 0.1, 0.10, 0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
         [0.1, 0.2, 0.1, 0.3, 0.1, 0.4, 0.1, 0.2, 0.1, 0.3, 0.1, 0.4],
         [0.1, 0.5, 0.1, 0.6, 0.1, 0.7, 0.1, 0.5, 0.1, 0.6, 0.1, 0.7],
         [0.1, 0.8, 0.1, 0.9, 0.1, 0.10, 0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
         [0.1, 0.2, 0.1, 0.3, 0.1, 0.4, 0.1, 0.2, 0.1, 0.3, 0.1, 0.4],
         [0.1, 0.5, 0.1, 0.6, 0.1, 0.7, 0.1, 0.5, 0.1, 0.6, 0.1, 0.7],
         [0.1, 0.8, 0.1, 0.9, 0.1, 0.10, 0.1, 0.8, 0.1, 0.9, 0.1, 0.10],
         [0.1, 0.2, 0.1, 0.3, 0.1, 0.4, 0.1, 0.2, 0.1, 0.3, 0.1, 0.4],
         [0.1, 0.5, 0.1, 0.6, 0.1, 0.7, 0.1, 0.5, 0.1, 0.6, 0.1, 0.7],
         [0.1, 0.8, 0.1, 0.9, 0.1, 0.10, 0.1, 0.8, 0.1, 0.9, 0.1, 0.10]]))
