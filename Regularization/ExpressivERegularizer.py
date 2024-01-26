import numpy as np
import pandas as pd

import torch

from pykeen.regularizers import Regularizer
from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory
from pykeen.trackers import ResultTracker

from class_resolver import HintOrType, OptionalKwargs

from typing import Optional, Mapping, Any
import re

from torch import Tensor

from Utils import preprocess_relations
from Regularization.ExpressivELRScheduler import ExpressivELRScheduler
from Regularization.ExpressivELogger import ExpressivELogger


class ExpressivERegularizer(Regularizer):

    __loss_functions_two_atoms: [int]
    __loss_aggregation: str
    __loss_limit: float
    __var_batch_size: int
    __const_batch_size: int
    __sampling_strategy: str
    __apply_rule_confidence: bool
    __tanh_map: bool
    __min_denom: float

    __iteration: int = 0

    __lr_scheduler: ExpressivELRScheduler
    __logger: ExpressivELogger

    __device: torch.device

    __factory: TriplesFactory
    __var_rules: pd.DataFrame
    __const_rules: pd.DataFrame

    __entity_weights: torch.FloatTensor

    def __init__(
            self,
            dataset: str,
            dataset_kwargs: Optional[Mapping[str, Any]],
            rules: str,
            rules_max_body_atoms: int = 2,
            var_rule_min_confidence: float = 0.1,
            const_rule_min_confidence: float = 0.5,
            loss_functions_two_atoms=None,
            loss_aggregation: str = "sum",
            loss_limit: float = 25.0,
            alpha: float = 1,
            min_alpha: float = 0,
            decay: str = "exponential",
            decay_rate: float = 5e-04,
            var_batch_size: int = None,
            const_batch_size: int = 0,
            sampling_strategy: str = "uniform",
            apply_rule_confidence = False,
            tanh_map: bool = True,
            min_denom: float = 0.5,
            tracked_rules=None,
            track_all_rules: bool = False,
            track_relation_params: bool = False,
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

        if var_rule_min_confidence > 1 or const_rule_min_confidence > 1:
            raise ValueError("Error: minimum rule confidence can't be greater than one!")

        if loss_functions_two_atoms is None:
            self.__loss_functions_two_atoms = [1, 2, 3, 4, 5, 6]
        else:
            self.__loss_functions_two_atoms = loss_functions_two_atoms

        if loss_aggregation != "sum" and loss_aggregation != "max":
            raise ValueError("Error: loss aggregation must be either sum or max!")
        self.__loss_aggregation = loss_aggregation

        if loss_limit <= 0:
            raise ValueError("Error: loss limit must be greater than 0!")
        self.__loss_limit = loss_limit

        self.__lr_scheduler = ExpressivELRScheduler(alpha, min_alpha, decay, decay_rate)

        if var_batch_size is not None and var_batch_size < 0:
            raise ValueError("Error: batch size must not be smaller than 0")
        self.__var_batch_size = var_batch_size
        if const_batch_size is None or const_batch_size < 0: # we can't train with all constant rules
            raise ValueError("Error: const batch size must not be smaller than 0 and can't be None")
        self.__const_batch_size = const_batch_size

        if sampling_strategy != "uniform" and sampling_strategy != "weighted":
            raise ValueError("Error: only uniform and weighted sampling strategy implemented!")
        self.__sampling_strategy = sampling_strategy

        self.__apply_rule_confidence = apply_rule_confidence
        self.__tanh_map = tanh_map
        self.__min_denom = min_denom

        self.__logger = ExpressivELogger(tanh_map, min_denom, tracked_rules, track_all_rules, track_relation_params, result_tracker, result_tracker_kwargs)

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
        rule_df = rule_df[rule_df['body_count'] <= rules_max_body_atoms]  # max body atoms
        # filter reflexive atoms
        rule_df = rule_df[rule_df['body'].apply(lambda atoms: all([self.__non_reflexive(atom) for atom in atoms]))]
        rule_df = rule_df[rule_df['head'].apply(self.__non_reflexive)]

        # add ids
        rule_df['body_ids'] = rule_df['body'].apply(self.__body_ids)
        rule_df['head_id'] = rule_df['head'].apply(self.__head_id)
        rule_df['ids'] = rule_df.apply(lambda x: set(x.body_ids).union([x.head_id]), axis=1)

        # no constants in head (in AnyBURL, there are never only consts in body)
        var_rule_df = rule_df[rule_df['head'].apply(self.__no_const_head)]
        var_rule_df = var_rule_df[var_rule_df['confidence'] >= var_rule_min_confidence]
        self.__var_rules = var_rule_df

        # constants in head (in AnyBURL, there are never only consts in body)
        const_rule_df = rule_df[rule_df['head'].apply(lambda atom: not self.__no_const_head(atom))]
        # rules with constants only in the head exists (i.e. r1(X,A) -> r2(X,const)
        # they are not handled as these rules have very low confidence and are only rarely justified
        const_rule_df = const_rule_df[const_rule_df['body'].apply(lambda atom: not self.__no_const_body(atom))]
        const_rule_df = const_rule_df[const_rule_df['confidence'] >= const_rule_min_confidence]
        const_rule_df['const_pos_body'] = const_rule_df['body'].apply(lambda body: self.__const_pos(body[0]))
        const_rule_df['const_pos_head'] = const_rule_df['head'].apply(self.__const_pos)
        const_rule_df['const_id_body'] = const_rule_df['body'].apply(lambda body: self.__const_id(body[0]))
        const_rule_df['const_id_head'] = const_rule_df['head'].apply(self.__const_id)
        self.__const_rules = const_rule_df

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Note: If lots of rules, split dataframe and parallelize. However, most likely upper limit (batch size) due to
        # complex backtracking.

        if x.size()[0] == len(self.__factory.entity_to_id):
            self.__entity_weights = x
            return torch.FloatTensor([0])

        self.__iteration += 1 # TODO: Sync with epochs
        self.__lr_scheduler.step()

        rules_loss = None
        var_rules = pd.DataFrame()
        const_rules = pd.DataFrame()

        if self.__var_batch_size is None:
            var_rules = self.__var_rules
        else:
            if self.__sampling_strategy == "uniform":
                var_rules = self.__var_rules.sample(self.__var_batch_size)
            elif self.__sampling_strategy == "weighted":
                var_rules = self.__var_rules.sample(self.__var_batch_size, weights="confidence")

        if self.__const_batch_size > 0:
            if self.__sampling_strategy == "uniform":
                const_rules = self.__const_rules.sample(self.__const_batch_size)
            elif self.__sampling_strategy == "weighted":
                const_rules = self.__const_rules.sample(self.__const_batch_size, weights="confidence")

        for idx, row in var_rules.iterrows():
            # use iterrows() as apply() + sum() throws error:
            # "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."

            rule_multiplier = row['confidence'] if self.__apply_rule_confidence else 1.0
            rule_loss = rule_multiplier * self.__compute_loss(row, x)

            if rules_loss is None:
                rules_loss = rule_loss
            else:
                rules_loss += rule_loss

            self.__logger.log_rule(idx, rule_loss, self.__iteration)

        relation_dim = x.size()[1] / 6
        const_intersections = 0.0
        for idx, row in const_rules.iterrows():
            # use iterrows() as apply() + sum() throws error:
            # "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."

            intersections, mask = self.__compute_const_relation_intersections(row, self.__entity_weights, x)
            # Generally, we either have 2 or 0 intersections per dimension
            const_intersections += float(torch.count_nonzero(mask)) / 2

            rule_multiplier = row['confidence'] if self.__apply_rule_confidence else 1.0
            rule_loss = rule_multiplier * self.__compute_const_loss(row, intersections, mask,
                                                                    self.__entity_weights, x)

            if rules_loss is None:
                rules_loss = rule_loss
            else:
                rules_loss += rule_loss

            self.__logger.log_rule(idx, rule_loss, self.__iteration)

        alpha = self.__lr_scheduler.alpha()
        const_body_satisfaction_percentage = const_intersections / float(len(const_rules) * relation_dim)

        self.__logger.log_weights(x, self.__iteration)
        self.__logger.log_alpha(alpha, self.__iteration)
        self.__logger.log_const_body_satisfaction(const_body_satisfaction_percentage, self.__iteration)
        self.__logger.log_rules(rule_loss, self.__iteration)

        return alpha * rules_loss

    def __non_reflexive(self, atom):
        # AnyBURL introduces a me_myself_i constant in reflexive rules
        # In general, we don't want to deal with reflexive rules in ExpressivE as AnyBURL doesn't differ between
        # general reflexiveness (reflexiveness is always fulfilled) and partial reflexiveness (reflexiveness only
        # fulfilled for a few constants)

        return 'me_myself' not in atom


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

    def __const_pos(self, atom: str) -> str:
        args = self.__extract_arguments(atom)

        x_pattern = re.compile('\(.{2}.*,[A-Z]\)') # TODO: Fix for FB15k
        y_pattern = re.compile('\([A-Z],.{2}.*\)') # TODO: Fix for FB15k

        x_result = x_pattern.match(args)
        y_result = y_pattern.match(args)

        if x_result is not None:
            return 'X'
        if y_result is not None:
            return 'Y'

        raise ValueError("Didn't detect constant in body!")

    def __body_ids(self, atoms: [str]) -> [int]:
        relations = list(map(self.__extract_relation, atoms))
        return self.__factory.relations_to_ids(relations)

    def __head_id(self, atom) -> int:
        relation = self.__extract_relation(atom)
        ids = list(self.__factory.relations_to_ids([relation]))
        return ids[0]

    def __const_id(self, atom) -> int:
        const = self.__extract_const(atom)

        # we need to work with entity_to_id as entities_to_ids() doesn't work, when we have integer entities
        mapping = self.__factory.entity_to_id

        try:
            const_converted = int(const)  # consts can have leading 0s, which must be removed
        except ValueError:
            const_converted = const

        return mapping[const_converted]

    def __extract_relation(self, atom: str) -> str:
        pattern = re.compile('[^(]*')
        regex_result = pattern.search(atom)
        relation = regex_result.group(0)

        return relation

    def __extract_const(self, atom: str) -> str:
        x_pos_pattern = re.compile('\(.{2}.*,')
        y_pos_pattern = re.compile(',.{2}.*\)')

        x_result = x_pos_pattern.search(atom)
        y_result = y_pos_pattern.search(atom)

        if x_result is not None:
            return x_result.group(0)[1:-1]
        if y_result is not None:
            return y_result.group(0)[1:-1]

        raise ValueError("Didn't detect constant in atom!")

    # noinspection PyTypeChecker
    def __compute_const_relation_intersections(self, rule, entities, relations) -> torch.FloatTensor:
        if len(rule['body']) > 1:
            print("Only const rules with body length 1 implemented!")

        # Intuition:
        # Each parallelogram is defined by four lines
        # We can evaluate each line at the const (= intersection between const and line)
        # Choose intersections, where score is below threshold
        #   - We have two score functions (<= d_h, <= d_t)
        #   - For intersection at d_h check second score below d_t (and vice versa)
        # Three possibilities:
        #   - No intersection: Return None
        #   - Intersection at corner: Return one intersection value
        #   - Intersections at top/bottom, left/right line: Return two intersection values

        const: torch.FloatTensor = entities[rule['const_id_body'], :]

        rel_weights = relations[rule['body_ids'][0], :]
        d_h, d_t, c_h, c_t, s_h, s_t = preprocess_relations(rel_weights, tanh_map=self.__tanh_map, min_denom=self.__min_denom)

        if rule['const_pos_body'] == 'X':
            head_equation_intersections_1 = (const - c_h - d_h) / s_t
            head_equation_intersections_2 = (const - c_h + d_h) / s_t

            tail_equation_intersections_1 = c_t + s_h * const - d_t
            tail_equation_intersections_2 = c_t + s_h * const + d_t

            head_equation_intersections_1_mask = torch.absolute(head_equation_intersections_1 - c_t - s_h * const) <= d_t
            head_equation_intersections_2_mask = torch.absolute(head_equation_intersections_2 - c_t - s_h * const) <= d_t

            tail_equation_intersections_1_mask = torch.absolute(const - c_h - s_t * tail_equation_intersections_1) <= d_h
            tail_equation_intersections_2_mask = torch.absolute(const - c_h - s_t * tail_equation_intersections_2) <= d_h

            intersections_list = [head_equation_intersections_1, head_equation_intersections_2,
                                  tail_equation_intersections_1, tail_equation_intersections_2]
            all_intersections = torch.stack(intersections_list, 1)

            mask_list = [head_equation_intersections_1_mask, head_equation_intersections_2_mask,
                         tail_equation_intersections_1_mask, tail_equation_intersections_2_mask]
            all_masks = torch.stack(mask_list, 1)

            return all_intersections, all_masks

        elif rule['const_pos_body'] == 'Y':
            head_equation_intersections_1 = c_h + s_t * const - d_h
            head_equation_intersections_2 = c_h + s_t * const + d_h

            tail_equation_intersections_1 = (const - c_t - d_t) / s_h
            tail_equation_intersections_2 = (const - c_t + d_t) / s_h

            head_equation_intersections_1_mask = torch.absolute(const - c_t - s_h * head_equation_intersections_1) <= d_t
            head_equation_intersections_2_mask = torch.absolute(const - c_t - s_h * head_equation_intersections_2) <= d_t

            tail_equation_intersections_1_mask = torch.absolute(tail_equation_intersections_1 - c_h - s_t * const) <= d_h
            tail_equation_intersections_2_mask = torch.absolute(tail_equation_intersections_2 - c_h - s_t * const) <= d_h

            intersections_list = [head_equation_intersections_1, head_equation_intersections_2,
                                  tail_equation_intersections_1, tail_equation_intersections_2]
            all_intersections = torch.stack(intersections_list, 1)

            mask_list = [head_equation_intersections_1_mask, head_equation_intersections_2_mask,
                         tail_equation_intersections_1_mask, tail_equation_intersections_2_mask]
            all_masks = torch.stack(mask_list, 1)

            return all_intersections, all_masks

        raise ValueError("Invalid constant position")

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
            return 0

        if body_args[0] == 'X,Y':
            # hierarchy: r(x,y) -> s(x,y) = r(x,y) and i(y,y) -> s(x,y)
            body_weights = weights[body_ids[0], :]
        else:
            # inversion: r(x,y) -> r(y,x) = r(x,y)
            # flip body relation weights
            body_weights = self.__flip_weights(weights[body_ids[0], :])
            body_weights = torch.flatten(body_weights)

        embedding_dim = int(len(body_weights) / 6)
        self_loop = torch.cat((torch.zeros(embedding_dim * 4, device=self.__device),
                               torch.ones(embedding_dim * 2, device=self.__device)))
        head_weights = weights[head_id, :]
        rule_weights = torch.stack((body_weights, self_loop, head_weights))
        return self.__general_composition_loss(rule_weights)

    def __compute_loss_two_atoms(self, body_args, head_args, body_ids, head_id, weights) -> torch.FloatTensor:
        if head_args != 'X,Y':
            # AnyBURl only supports acyclic rules of length 1 - should never land here
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
        return self.__general_composition_loss(rule_weights)

    def __compute_const_loss(self, rule, intersections, mask, entities, relations) -> torch.FloatTensor:
        if len(rule['body']) > 1:
            print("Only const rules with body length 1 implemented!")

        const: torch.FloatTensor = entities[rule['const_id_head'], :]
        const_stacked = torch.stack([const, const, const, const], 1)

        rel_weights = relations[rule['head_id'], :]
        d_h, d_t, c_h, c_t, s_h, s_t = preprocess_relations(rel_weights, tanh_map=self.__tanh_map, min_denom=self.__min_denom)
        d_h_stacked = torch.stack([d_h, d_h, d_h, d_h], 1)
        d_t_stacked = torch.stack([d_t, d_t, d_t, d_t], 1)
        c_h_stacked = torch.stack([c_h, c_h, c_h, c_h], 1)
        c_t_stacked = torch.stack([c_t, c_t, c_t, c_t], 1)
        s_h_stacked = torch.stack([s_h, s_h, s_h, s_h], 1)
        s_t_stacked = torch.stack([s_t, s_t, s_t, s_t], 1)

        # TODO: Use loss function from paper (scale with width of parallelogram)
        zeros = torch.zeros(intersections.size(), device=self.__device)
        if rule['const_pos_head'] == 'X':
            eq1_loss = torch.maximum(torch.absolute(const_stacked - c_h_stacked - s_t_stacked * intersections) - d_h_stacked, zeros)
            eq1_loss_mean = torch.mean(torch.masked_select(eq1_loss, mask))
            eq2_loss = torch.maximum(torch.absolute(intersections - c_t_stacked - s_h_stacked * const_stacked) - d_t_stacked, zeros)
            eq2_loss_mean = torch.mean(torch.masked_select(eq2_loss, mask))
        else:
            eq1_loss = torch.maximum(torch.absolute(intersections - c_h_stacked - s_t_stacked * const_stacked) - d_h_stacked, zeros)
            eq1_loss_mean = torch.mean(torch.masked_select(eq1_loss, mask))
            eq2_loss = torch.maximum(torch.absolute(const_stacked - c_t_stacked - s_h_stacked * intersections) - d_t_stacked, zeros)
            eq2_loss_mean = torch.mean(torch.masked_select(eq2_loss, mask))

        return eq1_loss_mean + eq2_loss_mean

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
        if len(weights.size()) == 1:
            num_weights = 1
            embedding_dim = int(weights.size()[0] / 6)
        else:
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

    def __general_composition_loss(self, weights, self_loop=False) -> torch.FloatTensor:
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

        # x = c3_h +/- d3_h + s3_t * y
        # y = c3_t +/- d3_t + s3_h * x

        # Corners = intersections of four equations
        # x = c3_h +/- d3_h + s3_t * c3_t +/- s3_t * d3_t / (1 - s3_t * s3_h)

        corner1_x = c3_h + d3_h + s3_t * c3_t + s3_t * d3_t / (1 - s3_t * s3_h)  # +,+
        corner1_y = c3_t + d3_t + s3_h * corner1_x

        corner2_x = c3_h + d3_h + s3_t * c3_t - s3_t * d3_t / (1 - s3_t * s3_h)  # +,-
        corner2_y = c3_t - d3_t + s3_h * corner2_x

        corner3_x = c3_h - d3_h + s3_t * c3_t + s3_t * d3_t / (1 - s3_t * s3_h)  # -,+
        corner3_y = c3_t + d3_t + s3_h * corner3_x

        corner4_x = c3_h - d3_h + s3_t * c3_t - s3_t * d3_t / (1 - s3_t * s3_h)  # -,-
        corner4_y = c3_t - d3_t + s3_h * corner4_x

        # Check each inequality with each corner point
        corner1_loss = self.__general_composition_corner_loss(corner1_x, corner1_y, rel_1, rel_2)
        corner2_loss = self.__general_composition_corner_loss(corner2_x, corner2_y, rel_1, rel_2)
        corner3_loss = self.__general_composition_corner_loss(corner3_x, corner3_y, rel_1, rel_2)
        corner4_loss = self.__general_composition_corner_loss(corner4_x, corner4_y, rel_1, rel_2)

        return corner1_loss + corner2_loss + corner3_loss + corner4_loss

    def __general_composition_corner_loss(self, corner_x, corner_y, rel_1, rel_2) -> Tensor:
        d1_h, d1_t, c1_h, c1_t, s1_h, s1_t = rel_1
        d2_h, d2_t, c2_h, c2_t, s2_h, s2_t = rel_2

        ones = torch.ones(corner_x.size(), device=self.__device)

        # TODO: Discuss aggregation function.
        # E.g.: Use dimension-wise maximum or sum
        if 1 in self.__loss_functions_two_atoms:
            eq1_loss = abs(corner_x - corner_y*s1_t*s2_t - c2_h*s1_t - c1_h) - d2_h*s1_t - d1_h
            eq1_loss = torch.mean(torch.clamp(eq1_loss, 0, self.__loss_limit))
        else:
            eq1_loss = 0

        if 2 in self.__loss_functions_two_atoms:
            eq2_loss = abs(corner_y*s2_t + c2_h - corner_x*s1_h - c1_t) - d1_t - d2_h
            eq2_loss = torch.mean(torch.clamp(eq2_loss, 0, self.__loss_limit))
        else:
            eq2_loss = 0

        if 3 in self.__loss_functions_two_atoms:
            eq3_loss = abs(corner_y - corner_x*s1_h*s2_h - c1_t*s2_h - c2_t) - d1_t*s2_h - d2_t
            eq3_loss = torch.mean(torch.clamp(eq3_loss, 0, self.__loss_limit))
        else:
            eq3_loss = 0

        # Note: s2_h/s1_t leads to explosion of loss term for some dimensions -> clamp
        # Other options: equal slopes variant, skip completely, scale log
        if 4 in self.__loss_functions_two_atoms:
            eq4_loss = abs(corner_y + (c1_h - corner_x)*s2_h/s1_t - c2_t) - d1_h*s2_h/s1_t - d2_t
            eq4_loss = torch.mean(torch.clamp(eq4_loss, 0, self.__loss_limit))
        else:
            eq4_loss = 0

        if 5 in self.__loss_functions_two_atoms:
            eq5_loss = abs(corner_x*(ones - s1_h*s1_t) - c1_t*s1_t - c1_h) - d1_t*s1_t - d1_h
            eq5_loss = torch.mean(torch.clamp(eq5_loss, 0, self.__loss_limit))
        else:
            eq5_loss = 0

        if 6 in self.__loss_functions_two_atoms:
            eq6_loss = abs(corner_y*(ones - s2_h*s2_t) - c2_h*s2_h - c2_t) - d2_h*s2_h - d2_t
            eq6_loss = torch.mean(torch.clamp(eq6_loss, 0, self.__loss_limit))
        else:
            eq6_loss = 0

        # Note: Prefer sum over max as eq4 tends to largest all the time (s2_h/s1_t)
        if self.__loss_aggregation == "sum":
            return eq1_loss + eq2_loss + eq3_loss + eq4_loss + eq5_loss + eq6_loss
        elif self.__loss_aggregation == "max":
            return max(eq1_loss, eq2_loss, eq3_loss, eq4_loss, eq5_loss, eq6_loss)

        # TODO: add assertion should never happen
        return 0


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
