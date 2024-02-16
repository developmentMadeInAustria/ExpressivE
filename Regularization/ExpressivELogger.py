
import torch
import numpy as np

from class_resolver import HintOrType, OptionalKwargs

from pykeen.trackers import ResultTracker, tracker_resolver
from pykeen.triples import TriplesFactory

from Utils import preprocess_relations


class ExpressivELogger:

    __tanh_map: bool
    __min_denom: float
    __tracked_rules: [int]
    __track_all_rules: bool
    __track_relation_params: bool
    __result_tracker: ResultTracker

    __relation_statistic_update_cycle: int
    __triples_factory: TriplesFactory
    __negative_triples_factory: TriplesFactory
    __entities: torch.FloatTensor = None
    __prev_entities: torch.FloatTensor = None
    __relations: torch.FloatTensor = None
    __prev_relations: torch.FloatTensor = None

    def __init__(self, tan_hmap: bool, min_denom: float, tracked_rules: list, track_all_rules: bool, track_relation_params: bool, relation_statistic_update_cycle: int, triples_factory: TriplesFactory, result_tracker: HintOrType[ResultTracker], result_tracker_kwargs: OptionalKwargs):

        self.__tanh_map = tan_hmap
        self.__min_denom = min_denom

        self.__tracked_rules = tracked_rules
        if self.__tracked_rules is None:
            self.__tracked_rules = []
        self.__track_all_rules = track_all_rules
        self.__track_relation_params = track_relation_params

        self.__relation_statistic_update_cycle = relation_statistic_update_cycle
        self.__triples_factory = triples_factory
        self.__generate_negative_samples()

        self.__result_tracker = tracker_resolver.make(query=result_tracker, pos_kwargs=result_tracker_kwargs)

    def update_entities(self, entities):
        self.__entities = entities

    def update_relations(self, relations):
        self.__relations = relations

    def log_rule(self, rule, loss, iteration):
        if self.__track_all_rules or rule in self.__tracked_rules:
            self.__result_tracker.log_metrics({"rule_{}_loss".format(rule): loss}, step=iteration)

    def log_rules(self, loss, iteration):
        self.__result_tracker.log_metrics({"rules_loss": loss}, step=iteration)

    def log_const_body_satisfaction(self, percentage, iteration):
        self.__result_tracker.log_metrics({"const_body_satisfaction_percentage": percentage}, step=iteration)

    def log_alpha(self, alpha, iteration):
        self.__result_tracker.log_metrics({"alpha": alpha}, step=iteration)

    def log_weights(self, weights, iteration):
        if not self.__track_relation_params:
            return

        for idx, rule in enumerate(weights):
            d_h, d_t, c_h, c_t, s_h, s_t = preprocess_relations(rule,
                                                                tanh_map=self.__tanh_map,
                                                                min_denom=self.__min_denom)

            self.__result_tracker.log_metrics({
                "rel_{}_dh".format(idx): torch.mean(d_h), "rel_{}_dt".format(idx): torch.mean(d_t),
                "rel_{}_ch".format(idx): torch.mean(c_h), "rel_{}_ct".format(idx): torch.mean(c_t),
                "rel_{}_sh".format(idx): torch.mean(s_h), "rel_{}_st".format(idx): torch.mean(s_t)
            }, step=iteration)

    def log_entity_relation_statistics(self, iteration):
        if self.__relation_statistic_update_cycle == -1 or (iteration % self.__relation_statistic_update_cycle) != 0:
            return

        if self.__entities is None or self.__relations is None:
            return

        if self.__prev_entities is not None and self.__prev_relations is not None:
            total_slope_change_head = 0
            total_slope_change_tail = 0
            total_distance_change_head = 0
            total_distance_change_tail = 0
            total_center_change_head = 0
            total_center_change_tail = 0

            for idx in range(0, self.__prev_relations.size()[0]):
                relation = self.__relations[idx]
                prev_relation = self.__prev_relations[idx]

                r_d_h, r_d_t, r_c_h, r_c_t, r_s_h, r_s_t = relation.tensor_split(6)
                pr_d_h, pr_d_t, pr_c_h, pr_c_t, pr_s_h, pr_s_t = prev_relation.tensor_split(6)

                slope_change_head = pr_s_h - r_s_h
                abs_slope_change_head = torch.abs(slope_change_head)
                average_slope_change_head = torch.mean(abs_slope_change_head)
                total_slope_change_head += float(average_slope_change_head)
                slope_change_tail = pr_s_t - r_s_t
                abs_slope_change_tail = torch.abs(slope_change_tail)
                average_slope_change_tail = torch.mean(abs_slope_change_tail)
                total_slope_change_tail += float(average_slope_change_tail)

                distance_change_head = pr_d_h - r_d_h
                abs_distance_change_head = torch.abs(distance_change_head)
                average_distance_change_head = torch.mean(abs_distance_change_head)
                total_distance_change_head += float(average_distance_change_head)
                distance_change_tail = pr_d_t - r_d_t
                abs_distance_change_tail = torch.abs(distance_change_tail)
                average_distance_change_tail = torch.mean(abs_distance_change_tail)
                total_distance_change_tail += float(average_distance_change_tail)

                center_change_head = pr_c_h - r_c_h
                abs_center_change_head = torch.abs(center_change_head)
                average_center_change_head = torch.mean(abs_center_change_head)
                total_center_change_head += float(average_center_change_head)
                center_change_tail = pr_c_t - r_c_t
                abs_center_change_tail = torch.abs(center_change_tail)
                average_center_change_tail = torch.mean(abs_center_change_tail)
                total_center_change_tail += float(average_center_change_tail)

                self.__result_tracker.log_metrics({
                    "rel_{}_slope_change_head".format(idx): average_slope_change_head,
                    "rel_{}_slope_change_tail".format(idx): average_slope_change_tail,
                    "rel_{}_distance_change_head".format(idx): average_distance_change_head,
                    "rel_{}_distance_change_tail".format(idx): average_distance_change_tail,
                    "rel_{}_center_change_head".format(idx): average_center_change_head,
                    "rel_{}_center_change_tail".format(idx): average_center_change_tail
                }, step=iteration)

            self.__result_tracker.log_metrics({
                "total_slope_change_head": total_slope_change_head,
                "total_slope_change_tail": total_slope_change_tail,
                "total_distance_change_head": total_distance_change_head,
                "total_distance_change_tail": total_distance_change_tail,
                "total_center_change_head": total_center_change_head,
                "total_center_change_tail": total_center_change_tail
            }, step=iteration)

            entity_change = self.__prev_entities - self.__entities
            abs_entity_change = torch.abs(entity_change)
            average_entity_change = torch.mean(abs_entity_change)

            self.__result_tracker.log_metrics({"avg_entity_change": float(average_entity_change)}, step=iteration)

        total_true_positives = 0  # counts number of dimensions, where positive triples are positive
        total_true_negatives = 0  # counts number of dimensions, where negative triples are negative
        total_false_positives = 0  # counts number of dimensions, where negative triples are positive
        num_dims = self.__relations.size()[1] / 3

        for idx in range(0, self.__relations.size()[0]):
            pos_mapped_triples = self.__triples_factory.mapped_triples
            relation_pos_triples = pos_mapped_triples[pos_mapped_triples[:, 1] == idx]
            neg_mapped_triples = self.__negative_triples_factory.mapped_triples
            relation_neg_triples = neg_mapped_triples[neg_mapped_triples[:, 1] == idx]

            num_pos_fulfilled_triples = self.__num_fulfilled_triples(relation_pos_triples, idx)
            total_true_positives += num_pos_fulfilled_triples
            num_neg_fulfilled_triples = self.__num_fulfilled_triples(relation_neg_triples, idx)
            total_true_negatives += num_neg_fulfilled_triples
            num_neg_unfulfilled_triples = (relation_neg_triples.size()[0] * num_dims) - num_neg_fulfilled_triples
            total_false_positives += num_neg_unfulfilled_triples

            if num_pos_fulfilled_triples + num_neg_fulfilled_triples > 0:
                rel_true_positive_rate = num_pos_fulfilled_triples / (num_pos_fulfilled_triples + num_neg_fulfilled_triples)
            else:
                rel_true_positive_rate = 0

            rel_sensitivity = num_pos_fulfilled_triples / (relation_pos_triples.size()[0] * num_dims)
            rel_specificity = num_neg_unfulfilled_triples / (relation_neg_triples.size()[0] * num_dims)

            self.__result_tracker.log_metrics({
                "rel_{}_true_positive_rate".format(idx): rel_true_positive_rate,
                "rel_{}_sensitivity".format(idx): rel_sensitivity,
                "rel_{}_specificity".format(idx): rel_specificity
            }, step=iteration)

        if total_true_positives + total_false_positives > 0:
            total_true_positive_rate = total_true_positives / (total_true_positives + total_false_positives)
        else:
            total_true_positive_rate = 0

        total_sensitivity = total_true_positives / (self.__triples_factory.num_triples * num_dims)
        total_specificity = total_true_negatives / (self.__triples_factory.num_triples * num_dims)

        self.__result_tracker.log_metrics({
            "total_true_positive_rate": total_true_positive_rate,
            "total_sensitivity": total_sensitivity,
            "total_specificity": total_specificity
        }, step=iteration)

        self.__prev_entities = self.__entities
        self.__prev_relations = self.__relations

    def __generate_negative_samples(self):
        entities = np.unique(self.__triples_factory.mapped_triples[:, 0])
        negative_triples: torch.LongTensor = torch.empty(self.__triples_factory.num_triples, 3, dtype=torch.long)

        negative_triples[:, 0] = torch.from_numpy(np.random.choice(entities, size=self.__triples_factory.num_triples))
        negative_triples[:, 1] = self.__triples_factory.mapped_triples[:, 1]
        negative_triples[:, 2] = torch.from_numpy(np.random.choice(entities, size=self.__triples_factory.num_triples))

        self.__negative_triples_factory = TriplesFactory(negative_triples,
                                                         self.__triples_factory.entity_to_id,
                                                         self.__triples_factory.relation_to_id,
                                                         self.__triples_factory.create_inverse_triples,
                                                         self.__triples_factory.metadata,
                                                         self.__triples_factory.num_entities,
                                                         self.__triples_factory.num_relations)

    def __num_fulfilled_triples(self, triples, relation) -> int:
        h = self.__entities[triples[:, 0]]
        t = self.__entities[triples[:, 2]]
        d, c, s = self.__relations[relation].tensor_split(3)

        ht = torch.cat(torch.broadcast_tensors(h, t), dim=-1)
        th = torch.cat(torch.broadcast_tensors(t, h), dim=-1)

        contextualized_pos = torch.abs(ht - c - torch.mul(s, th))
        is_entity_pair_within_para = torch.le(contextualized_pos, d)

        return torch.count_nonzero(is_entity_pair_within_para)
