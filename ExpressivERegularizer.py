
import pandas as pd

import torch

from pykeen.regularizers import Regularizer
from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory

from typing import Optional, Mapping, Any
import re

from Utils import preprocess_relations


class ExpressivERegularizer(Regularizer):

    __factory: TriplesFactory
    __rules: pd.DataFrame
    __tanh_map: bool
    __min_denom: float

    def __init__(
            self,
            dataset: str,
            dataset_kwargs: Optional[Mapping[str, Any]],
            rules: str,
            rules_max_body_atoms: int,
            rule_min_confidence: float,
            tanh_map: bool = True,
            min_denom: float = 0.5,
            **kwargs
    ) -> None:
        kwargs['apply_only_once'] = True
        super().__init__(**kwargs)

        self.__tanh_map = tanh_map
        self.__min_denom = min_denom
        # TODO: Add argument regularizer weight
        # TODO: Check arguments, if rules_max_body_atoms larger than currently implemented, raise error

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
        # apply() + sum() throws error
        # "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."

        # TODO: If lots of rules, split dataframe and parallelize
        rules_loss = 0
        for idx, row in self.__rules.iterrows():
            rules_loss += self.__compute_loss(row, x)
        return torch.FloatTensor(rules_loss)

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

    def __compute_loss(self, rule, weights) -> float:
        body_args = map(self.__extract_arguments, rule['body'])
        body_args = list(map(lambda args: args[1:-1], body_args))
        head_args = self.__extract_arguments(rule['head'])
        head_args = head_args[1:-1]

        # TODO: Implement body_count == 0

        if rule['body_count'] == 1:
            return self.__compute_loss_one_atom(body_args, head_args, rule['body_ids'], rule['head_id'], weights)
        elif rule['body_count'] == 2:
            return self.__compute_loss_two_atoms(body_args, head_args, rule['body_ids'], rule['head_id'], weights)

        return 0

    # TODO: Move to separate class
    def __compute_loss_one_atom(self, body_args, head_args, body_ids, head_id, weights) -> float:
        if head_args == 'X,Y':
            # hierarchy: r(x,y) -> s(x,y) = r(x,y) and i(y,y) -> s(x,y)
            body_weights = weights[body_ids[0], :]
            embedding_dim = int(len(body_weights) / 6)
            self_loop = torch.cat((torch.zeros(embedding_dim*4), torch.ones(embedding_dim*2)))
            head_weights = weights[head_id, :]
            rule_weights = torch.stack((body_weights, self_loop, head_weights))
            return self.general_composition_loss(rule_weights)
        else:
            # inversion: r(x,y) -> r(y,x) = r(x,y) and i(y,y) -> r(y,x)
            print("Loss for inversion rules not implemented yet")
            return 0

    def __compute_loss_two_atoms(self, body_args, head_args, body_ids, head_id, weights) -> float:
        chain_order = self.__compute_chain_order(body_args, [], 'X')

        if all(chain_order) and head_args == 'X,Y':
            body_indices = torch.tensor(body_ids)
            body_weights = torch.index_select(weights, 0, body_indices)
            head_weights = weights[head_id, :]
            head_weights = torch.reshape(head_weights, (1, -1))
            rule_weights = torch.cat((body_weights, head_weights), dim=0)
            return self.general_composition_loss(rule_weights)
        else:
            print("Loss function only for chained, connected rules without inverted variables implemented.")
            return 0

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

    def general_composition_loss(self, weights, self_loop=False) -> float:
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

    def general_composition_corner_loss(self, corner_x, corner_y, rel_1, rel_2) -> float:
        d1_h, d1_t, c1_h, c1_t, s1_h, s1_t = rel_1
        d2_h, d2_t, c2_h, c2_t, s2_h, s2_t = rel_2

        zero_loss = torch.zeros(corner_x.size())
        ones = torch.ones(corner_x.size())

        eq1_loss = abs(corner_x - corner_y*s1_t*s2_t - c2_h*s1_t - c1_h) - d2_h*s1_t - d1_h
        eq1_loss = torch.sum(torch.maximum(zero_loss, eq1_loss))

        eq2_loss = abs(corner_y*s2_t + c2_h - corner_x*s1_h - c1_t) - d1_t - d2_h
        eq2_loss = torch.sum(torch.maximum(zero_loss, eq2_loss))

        eq3_loss = abs(corner_y - corner_x*s1_h*s2_h - c1_t*s2_h - c2_t) - d1_t*s2_h - d2_t
        eq3_loss = torch.sum(torch.maximum(zero_loss, eq3_loss))

        eq4_loss = abs(corner_y + (c1_h - corner_x)*s2_h/s1_t - c2_t) - d1_h*s2_h/s1_t - d2_t
        eq4_loss = torch.sum(torch.maximum(zero_loss, eq4_loss))

        eq5_loss = abs(corner_x*(ones - s1_h*s1_t) - c1_t*s1_t - c1_h) - d1_t*s1_t - d1_h
        eq5_loss = torch.sum(torch.maximum(zero_loss, eq5_loss))

        eq6_loss = abs(corner_y*(ones - s2_h*s2_t) - c2_h*s2_h - c2_t) - d2_h*s2_h - d2_t
        eq6_loss = torch.sum(torch.maximum(zero_loss, eq6_loss))

        return eq1_loss + eq2_loss + eq3_loss + eq4_loss + eq5_loss + eq6_loss


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
