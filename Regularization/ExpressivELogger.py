
import torch

from class_resolver import HintOrType, OptionalKwargs

from pykeen.trackers import ResultTracker, tracker_resolver

from Utils import preprocess_relations


class ExpressivELogger:

    __tanh_map: bool
    __min_denom: float
    __tracked_rules: [int]
    __track_all_rules: bool
    __track_relation_params: bool
    __result_tracker: ResultTracker

    def __init__(self, tan_hmap: bool, min_denom: float, tracked_rules: list, track_all_rules: bool, track_relation_params: bool, result_tracker: HintOrType[ResultTracker], result_tracker_kwargs: OptionalKwargs):

        self.__tanh_map = tan_hmap
        self.__min_denom = min_denom

        self.__tracked_rules = tracked_rules
        if self.__tracked_rules is None:
            self.__tracked_rules = []
        self.__track_all_rules = track_all_rules
        self.__track_relation_params = track_relation_params
        self.__result_tracker = tracker_resolver.make(query=result_tracker, pos_kwargs=result_tracker_kwargs)

    def log_rule(self, rule, loss, iteration):
        if self.__track_all_rules or rule in self.__tracked_rules:
            self.__result_tracker.log_metrics({"rule_{}_loss".format(rule): loss}, step=iteration)

    def log_rules(self, loss, iteration):
        self.__result_tracker.log_metrics({"rules_loss": loss}, step=iteration)

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

