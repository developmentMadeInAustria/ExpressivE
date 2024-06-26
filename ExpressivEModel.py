from typing import ClassVar, Mapping, Any
from class_resolver import HintOrType, OptionalKwargs

import torch
from pykeen.losses import NSSALoss
from pykeen.models import ERModel
from pykeen.nn.representation import Embedding
from pykeen.regularizers import Regularizer

from Ablations import EqSlopesInteraction, FunctionalInteraction, NoCenterInteraction, OneBandInteraction
from ExpressivEInteraction import ExpressivEInteraction


class ExpressivE(ERModel):
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=9, high=11, scale='power', base=2),
        p=dict(type=int, low=1, high=2),
        min_denom=dict(type=float, low=2e-1, high=8e-1, step=1e-1)
    )

    loss_default = NSSALoss
    loss_default_kwargs = dict(margin=3, adversarial_temperature=2.0, reduction='sum')

    def __init__(
            self,
            embedding_dim: int = 50,
            p: int = 2,
            min_denom=0.5,
            tanh_map=True,
            interactionMode='baseExpressivE',
            regularizer: HintOrType[Regularizer] = None,
            regularizer_kwargs: OptionalKwargs = None,
            **kwargs,
    ) -> None:

        triples_factory = kwargs['triples_factory']

        if min_denom > 0 and interactionMode != 'baseExpressivE':
            raise Exception('The specified ExpressivE variant does not use the <min_denom> argument.\\'
                            'Please set <min_denom>=0.')

        if interactionMode == 'baseExpressivE':
            print('<<< Base ExpressivE >>>')
            # Base ExpressivE does not constrain its parameters.

            super().__init__(
                interaction=ExpressivEInteraction(p=p, min_denom=min_denom, tanh_map=tanh_map),
                entity_representations=Embedding(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                ),
                relation_representations=Embedding(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=6 * embedding_dim,
                ),  # d_h, d_t, c_h, c_t, s_h, s_t
                **kwargs,
            )

        elif interactionMode == 'functional':
            print('<<< Functional ExpressivE >>>')
            # Functional ExpressivE drops the width vectors.

            super().__init__(
                interaction=FunctionalInteraction(p=p, tanh_map=tanh_map),
                entity_representations=Embedding(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                ),
                relation_representations=Embedding(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=4 * embedding_dim,
                ),  # c_h, c_t, s_h, s_t
                **kwargs,
            )

        elif interactionMode == 'eqSlopes':
            print('<<< ExpressivE with equal slopes >>>')
            # EqSlopes ExpressivE sets all slope vectors to be equal.

            super().__init__(
                interaction=EqSlopesInteraction(p=p, embedding_dim=embedding_dim, tanh_map=tanh_map),
                entity_representations=Embedding(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                ),
                relation_representations=Embedding(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=4 * embedding_dim,
                ),  # d_h, d_t, c_h, c_t
                **kwargs,
            )

            self.global_slope = torch.nn.Parameter(self.interaction.s, requires_grad=True).cuda()

        elif interactionMode == 'noCenter':
            print('<<< NoCenter ExpressivE >>>')
            # Functional ExpressivE drops the center vectors.

            super().__init__(
                interaction=NoCenterInteraction(p=p, tanh_map=tanh_map),
                entity_representations=Embedding(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                ),
                relation_representations=Embedding(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=4 * embedding_dim,
                ),  # d_h, d_t, s_h, s_t
                **kwargs,
            )

        elif interactionMode == 'oneBand':
            print('<<< One-Band ExpressivE >>>')
            # One-Band ExpressivE uses only one band instead of two bands.

            super().__init__(
                interaction=OneBandInteraction(p=p, tanh_map=tanh_map),
                entity_representations=Embedding(
                    num_embeddings=triples_factory.num_entities,
                    embedding_dim=embedding_dim,
                ),
                relation_representations=Embedding(
                    num_embeddings=triples_factory.num_relations,
                    embedding_dim=3 * embedding_dim,
                ),  # d, c, s
                **kwargs,
            )

        else:
            raise Exception('<<< Interaction Mode unkown! >>>')

        if regularizer is not None:
            self.append_weight_regularizer(
                parameter=self.entity_representations[0].parameters(),
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs
            )

            self.append_weight_regularizer(
                parameter=self.relation_representations[0].parameters(),
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            )

