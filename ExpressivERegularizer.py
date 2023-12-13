
import pandas as pd

import torch

from pykeen.regularizers import Regularizer
from pykeen.datasets import get_dataset
from pykeen.triples import TriplesFactory

from typing import Optional, Mapping, Any
import re


class ExpressivERegularizer(Regularizer):

    __factory: TriplesFactory
    __rules: pd.DataFrame

    def __init__(
            self,
            dataset: str,
            dataset_kwargs: Optional[Mapping[str, Any]],
            rules: str,
            rules_max_body_atoms: int,
            rule_min_confidence: float,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

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
        rule_df['ids'] = rule_df.apply(lambda x: x.body_ids.union([x.head_id]), axis=1)

        self.__rules = rule_df

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.FloatTensor([0])

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

    def __body_ids(self, atoms: [str]) -> set:
        relations = list(map(self.__extract_relation, atoms))
        ids = self.__factory.relations_to_ids(relations)
        return set(ids)

    def __head_id(self, atom) -> int:
        relation = self.__extract_relation(atom)
        ids = list(self.__factory.relations_to_ids([relation]))
        return ids[0]

    def __extract_relation(self, atom: str) -> str:
        pattern = re.compile('[^(]*')
        regex_result = pattern.search(atom)
        relation = regex_result.group(0)

        return relation

if __name__ == '__main__':
    reg = ExpressivERegularizer("WN18RR", dataset_kwargs={}, rules="../Rules/WN18RR-Max5-Run1/WN18RR-1000", rules_max_body_atoms=3, rule_min_confidence=0.1)
