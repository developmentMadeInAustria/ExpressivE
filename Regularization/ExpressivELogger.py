
import torch
import numpy as np

from class_resolver import HintOrType, OptionalKwargs

from pykeen.trackers import ResultTracker, tracker_resolver
from pykeen.triples import TriplesFactory

from Utils import preprocess_relations


class ExpressivELogger:

    __tanh_map: bool
    __min_denom: float
    __p: int
    __tracked_rules: [int]
    __track_all_rules: bool
    __track_relation_params: bool
    __result_tracker: ResultTracker

    __relation_statistic_update_cycle: int
    __result_statistics_update_cycle: int
    __max_metrics_dimension: int
    __triples_factory: TriplesFactory
    __negative_triples_factory: TriplesFactory
    __entities: torch.FloatTensor = None
    __prev_entities: torch.FloatTensor = None
    __relations: torch.FloatTensor = None
    __prev_relations: torch.FloatTensor = None

    def __init__(self, tan_hmap: bool, min_denom: float, p: int, tracked_rules: list, track_all_rules: bool, track_relation_params: bool, relation_statistic_update_cycle: int, result_statistics_update_cycle: int, max_metrics_dimension: int, triples_factory: TriplesFactory, result_tracker: HintOrType[ResultTracker], result_tracker_kwargs: OptionalKwargs):

        self.__tanh_map = tan_hmap
        self.__min_denom = min_denom
        self.__p = p

        self.__tracked_rules = tracked_rules
        if self.__tracked_rules is None:
            self.__tracked_rules = []
        self.__track_all_rules = track_all_rules
        self.__track_relation_params = track_relation_params

        self.__relation_statistic_update_cycle = relation_statistic_update_cycle
        self.__result_statistics_update_cycle = result_statistics_update_cycle
        self.__max_metrics_dimension = max_metrics_dimension
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

    def log_const_head_satisfaction(self, percentage, iteration):
        self.__result_tracker.log_metrics({"const_head_satisfaction_percentage": percentage}, step=iteration)

    def log_const_head_per_body_satisfaction(self, percentage, iteration):
        self.__result_tracker.log_metrics({"const_head_per_body_satisfaction_percentage": percentage}, step=iteration)

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
            if self.__prev_entities is None or self.__prev_relations is None:
                self.__prev_entities = self.__entities.clone()
                self.__prev_relations = self.__relations.clone()
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
            total_corner_point_change = 0

            total_diam1 = 0
            total_diam2 = 0
            total_area = 0

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

                current_corner_points = self.__calculate_corner_points(r_d_h, r_d_t, r_c_h, r_c_t, r_s_h, r_s_t)
                prev_corner_points = self.__calculate_corner_points(pr_d_h, pr_d_t, pr_c_h, pr_c_t, pr_s_h, pr_s_t)
                corner_point_change = prev_corner_points - current_corner_points
                abs_corner_point_change = torch.abs(corner_point_change)
                average_corner_point_change = torch.mean(abs_corner_point_change)
                total_corner_point_change += average_corner_point_change

                rel_diam1, rel_diam2, rel_area = self.__calculate_diameters_and_area(current_corner_points[0],
                                                                                     current_corner_points[1],
                                                                                     current_corner_points[2],
                                                                                     current_corner_points[3],
                                                                                     current_corner_points[4],
                                                                                     current_corner_points[5],
                                                                                     current_corner_points[6],
                                                                                     current_corner_points[7])
                total_diam1 += rel_diam1
                total_diam2 += rel_diam2
                total_area += rel_area

                self.__result_tracker.log_metrics({
                    "rel_{}_slope_change_head".format(idx): average_slope_change_head,
                    "rel_{}_slope_change_tail".format(idx): average_slope_change_tail,
                    "rel_{}_distance_change_head".format(idx): average_distance_change_head,
                    "rel_{}_distance_change_tail".format(idx): average_distance_change_tail,
                    "rel_{}_center_change_head".format(idx): average_center_change_head,
                    "rel_{}_center_change_tail".format(idx): average_center_change_tail,
                    "rel_{}_corner_point_change".format(idx): average_corner_point_change,
                    "rel_{}_diam1".format(idx): rel_diam1,
                    "rel_{}_diam2".format(idx): rel_diam2,
                    "rel_{}_area".format(idx): rel_area
                }, step=iteration)

            self.__result_tracker.log_metrics({
                "total_slope_change_head": total_slope_change_head,
                "total_slope_change_tail": total_slope_change_tail,
                "total_distance_change_head": total_distance_change_head,
                "total_distance_change_tail": total_distance_change_tail,
                "total_center_change_head": total_center_change_head,
                "total_center_change_tail": total_center_change_tail,
                "total_corner_point_change": total_corner_point_change,
                "total_diam1": total_diam1,
                "total_diam2": total_diam2,
                "total_area": total_area
            }, step=iteration)

            entity_change = self.__prev_entities - self.__entities
            abs_entity_change = torch.abs(entity_change)
            average_entity_change = torch.mean(abs_entity_change)

            self.__result_tracker.log_metrics({"avg_entity_change": float(average_entity_change)}, step=iteration)

        self.__prev_entities = self.__entities.clone()
        self.__prev_relations = self.__relations.clone()

    def log_result_statistics(self, iteration):
        if self.__relation_statistic_update_cycle == -1 or (iteration % self.__result_statistics_update_cycle) != 0:
            return

        if self.__entities is None or self.__relations is None:
            return

        total_true_positives = 0  # counts number of dimensions, where positive triples are positive
        total_true_negatives = 0  # counts number of dimensions, where negative triples are negative
        total_false_positives = 0  # counts number of dimensions, where negative triples are positive
        mean_true_score = 0  # the mean score of true triples
        mean_false_score = 0  # the mean score false triples
        num_dims = self.__relations.size()[1] / 3

        dimension_total_true_positives = np.zeros(self.__max_metrics_dimension + 1)
        dimension_total_true_negatives = np.zeros(self.__max_metrics_dimension + 1)
        dimension_total_false_positives = np.zeros(self.__max_metrics_dimension + 1)
        dimension_mean_true_score = np.zeros(self.__max_metrics_dimension + 1)
        dimension_mean_false_score = np.zeros(self.__max_metrics_dimension + 1)

        for idx in range(0, self.__relations.size()[0]):
            pos_mapped_triples = self.__triples_factory.mapped_triples
            relation_pos_triples = pos_mapped_triples[pos_mapped_triples[:, 1] == idx]
            neg_mapped_triples = self.__negative_triples_factory.mapped_triples
            relation_neg_triples = neg_mapped_triples[neg_mapped_triples[:, 1] == idx]

            num_pos_fulfilled_triples, pos_triples_score = self.__num_fulfilled_triples(relation_pos_triples, idx)
            total_true_positives += num_pos_fulfilled_triples
            mean_true_score += pos_triples_score * relation_pos_triples.size()[0] / pos_mapped_triples.size()[0]
            num_neg_fulfilled_triples, neg_triples_score = self.__num_fulfilled_triples(relation_neg_triples, idx)
            total_false_positives += num_neg_fulfilled_triples
            mean_false_score += neg_triples_score * relation_neg_triples.size()[0] / neg_mapped_triples.size()[0]
            # * 2 because of head and tail check
            num_neg_unfulfilled_triples = (relation_neg_triples.size()[0] * num_dims * 2) - num_neg_fulfilled_triples
            total_true_negatives += num_neg_unfulfilled_triples

            if self.__max_metrics_dimension > 0:
                for dim in range(0, self.__max_metrics_dimension):
                    dim_num_pos_fulfilled_triples, dim_pos_score = self.__num_fulfilled_triples(relation_pos_triples, idx, dim)
                    dimension_total_true_positives[dim] += dim_num_pos_fulfilled_triples
                    dimension_mean_true_score[dim] += dim_pos_score * relation_pos_triples.size()[0] / pos_mapped_triples.size()[0]
                    dim_num_neg_fulfilled_triples, dim_neg_score = self.__num_fulfilled_triples(relation_neg_triples, idx, dim)
                    dimension_total_false_positives[dim] += dim_num_neg_fulfilled_triples
                    dimension_mean_false_score[dim] += dim_neg_score * relation_neg_triples.size()[0] / neg_mapped_triples.size()[0]
                    dim_num_neg_unfulfilled_triples = relation_neg_triples.size()[0] * 2 - dim_num_neg_fulfilled_triples
                    dimension_total_true_negatives[dim] += dim_num_neg_unfulfilled_triples

            if num_pos_fulfilled_triples + num_neg_fulfilled_triples > 0:
                rel_true_positive_rate = num_pos_fulfilled_triples / (
                            num_pos_fulfilled_triples + num_neg_fulfilled_triples)
            else:
                rel_true_positive_rate = 0

            rel_sensitivity = num_pos_fulfilled_triples / (relation_pos_triples.size()[0] * num_dims * 2)
            rel_specificity = num_neg_unfulfilled_triples / (relation_neg_triples.size()[0] * num_dims * 2)

            self.__result_tracker.log_metrics({
                "rel_{}_true_positive_rate".format(idx): rel_true_positive_rate,
                "rel_{}_sensitivity".format(idx): rel_sensitivity,
                "rel_{}_specificity".format(idx): rel_specificity,
                "rel_{}_pos_score".format(idx): pos_triples_score,
                "rel_{}_neg_score".format(idx): neg_triples_score,
                "rel_{}_pos_neg_score_diff".format(idx): pos_triples_score - neg_triples_score
            }, step=iteration)

        if total_true_positives + total_false_positives > 0:
            total_true_positive_rate = total_true_positives / (total_true_positives + total_false_positives)
        else:
            total_true_positive_rate = 0

        total_sensitivity = total_true_positives / (self.__triples_factory.num_triples * num_dims * 2)
        total_specificity = total_true_negatives / (self.__triples_factory.num_triples * num_dims * 2)

        self.__result_tracker.log_metrics({
            "total_true_positive_rate": total_true_positive_rate,
            "total_sensitivity": total_sensitivity,
            "total_specificity": total_specificity,
            "total_mean_pos_score": mean_true_score,
            "total_mean_neg_score": mean_false_score,
            "total_mean_score_diff": mean_true_score - mean_false_score
        }, step=iteration)

        if self.__max_metrics_dimension > 0:
            for dim in range(0, self.__max_metrics_dimension):
                if dimension_total_true_positives[dim] + dimension_total_false_positives[dim] > 0:
                    dim_total_true_positive_rate = dimension_total_true_positives[dim] / (dimension_total_true_positives[dim] + dimension_total_false_positives[dim])
                else:
                    dim_total_true_positive_rate = 0

                dim_total_sensitivity = dimension_total_true_positives[dim] / (self.__triples_factory.num_triples * 2)
                dim_total_specificity = dimension_total_true_negatives[dim] / (self.__triples_factory.num_triples * 2)

                self.__result_tracker.log_metrics({
                    "dim_{}_total_true_positive_rate".format(dim): dim_total_true_positive_rate,
                    "dim_{}_sensitivity".format(dim): dim_total_sensitivity,
                    "dim_{}_specificity".format(dim): dim_total_specificity,
                    "dim_{}_mean_pos_score".format(dim): dimension_mean_true_score[dim],
                    "dim_{}_mean_neg_score".format(dim): dimension_mean_false_score[dim],
                    "dim_{}_mean_score_diff".format(dim): dimension_mean_true_score[dim] - dimension_mean_false_score[dim]
                }, step=iteration)

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

    def __calculate_corner_points(self, d_h, d_t, c_h, c_t, s_h, s_t):
        # How to calculate corner points?
        # Calculate intersection of lines
        # Example Corner 1 (see also GeoGebra):
        # y = c_t + d_t + x * s_h
        # x = c_h + d_h + y * s_t
        # Then, only change sign of d_t and d_h
        # Note, corners go round in a circle (from 1-4)

        corner1_x = (c_h + d_h + s_t * c_t + s_t * d_t) / (1 - s_t * s_h)  # + d_h, + d_t
        corner1_y = c_t + d_t + s_h * corner1_x

        corner2_x = (c_h + d_h + s_t * c_t - s_t * d_t) / (1 - s_t * s_h)  # + d_h, - d_t
        corner2_y = c_t - d_t + s_h * corner2_x

        corner3_x = (c_h - d_h + s_t * c_t - s_t * d_t) / (1 - s_t * s_h)  # - d_h, - d_t
        corner3_y = c_t - d_t + s_h * corner3_x

        corner4_x = (c_h - d_h + s_t * c_t + s_t * d_t) / (1 - s_t * s_h)  # - d_h, + d_t
        corner4_y = c_t + d_t + s_h * corner4_x

        return torch.stack([corner1_x, corner1_y, corner2_x, corner2_y, corner3_x, corner3_y, corner4_x, corner4_y])

    def __calculate_diameters_and_area(self, corner1_x, corner1_y, corner2_x, corner2_y, corner3_x, corner3_y, corner4_x, corner4_y) -> (float, float, float):
        # opposite corners: 1 & 3, 2 & 4

        # calculate side lengths of parallelogram
        a = torch.sqrt(torch.pow(corner1_x - corner2_x, 2) + torch.pow(corner1_y - corner2_y, 2))
        b = torch.sqrt(torch.pow(corner2_x - corner3_x, 2) + torch.pow(corner2_y - corner3_y, 2))

        # calculate diameters
        diam1 = torch.sqrt(torch.pow(corner1_x - corner3_x, 2) + torch.pow(corner1_y - corner3_y, 2))
        diam2 = torch.sqrt(torch.pow(corner2_x - corner4_x, 2) + torch.pow(corner2_y - corner4_y, 2))

        # Heron's formula for area
        semi_perimeter = (a + b + diam1) / 2
        area = 2 * torch.sqrt(semi_perimeter * (semi_perimeter - a) * (semi_perimeter - b) * (semi_perimeter - diam1))
        return torch.mean(diam1), torch.mean(diam2), torch.mean(area)

    def __num_fulfilled_triples(self, triples, relation, dimension=None) -> (int, any):
        if dimension is None:
            h = self.__entities[triples[:, 0]]
            t = self.__entities[triples[:, 2]]
            d, c, s = self.__relations[relation].tensor_split(3)

            ht = torch.cat(torch.broadcast_tensors(h, t), dim=-1)
            th = torch.cat(torch.broadcast_tensors(t, h), dim=-1)
        else:
            h = self.__entities[triples[:, 0]][:, dimension]
            t = self.__entities[triples[:, 2]][:, dimension]

            ht = torch.stack(torch.broadcast_tensors(h, t), dim=1)
            th = torch.stack(torch.broadcast_tensors(t, h), dim=1)

            d_h, d_t, c_h, c_t, s_h, s_t = self.__relations[relation].tensor_split(6)
            d = torch.tensor([d_h[dimension], d_t[dimension]], device=d_h.device)
            c = torch.tensor([c_h[dimension], c_t[dimension]], device=c_h.device)
            s = torch.tensor([s_h[dimension], s_t[dimension]], device=s_h.device)

        contextualized_pos = torch.abs(ht - c - torch.mul(s, th))
        is_entity_pair_within_para = torch.le(contextualized_pos, d)

        w = 2 * d + 1

        k = torch.mul(0.5 * (w - 1), (w - 1 / w))
        dist = torch.mul(contextualized_pos, w) - k

        dist[is_entity_pair_within_para] = torch.div(contextualized_pos, w)[is_entity_pair_within_para]

        num_fulfilled_triples = int(torch.count_nonzero(is_entity_pair_within_para))
        score = torch.mean(-dist.norm(p=self.__p, dim=-1))

        return num_fulfilled_triples, score
