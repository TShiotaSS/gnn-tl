"""
The core m3gnet model
"""
import json
import logging
import os
import urllib.request
from typing import List, Optional
import pandas as pd
# import numpy as np
import os

import numpy as np
import tensorflow as tf

from m3gnet.graph import Index, RadiusCutoffGraphConverter, tf_compute_distance_angle
from m3gnet.layers import (
    MLP,
    AtomReduceState,
    AtomRef,
    BaseAtomRef,
    ConcatAtoms,
    ConcatBondAtomState,
    GatedAtomUpdate,
    GatedMLP,
    GraphFeaturizer,
    GraphNetworkLayer,
    GraphUpdateFunc,
    MultiFieldReadout,
    Pipe,
    ReduceReadOut,
    Set2Set,
    SphericalBesselWithHarmonics,
    StateNetwork,
    ThreeDInteraction,
    WeightedReadout,
    polynomial,
)
from m3gnet.utils import register_plain

from ._base import GraphModelMixin

logger = logging.getLogger(__file__)

CWD = os.path.dirname(os.path.abspath(__file__))

"""
# Pre-trained models naming guidelines

To ensure clarity on the training data on the models, the naming convention for pre-trained models should be
<source>-<date in YYYY.MM.DD format>-E(F)(S), where the E, F and S denotes energies, forces and stresses respectively.
For example, MP-2021.2.8-EFS denotes a potential trained on Materials Project energies, forces and stresses as of
2021.2.8.
"""
MODEL_NAME = "m3gnet"

MODEL_PATHS = {"MP-2021.2.8-EFS": os.path.join(CWD, "..", "..", "pretrained", "MP-2021.2.8-EFS")}

MODEL_FILES = {
    "MP-2021.2.8-EFS": ["checkpoint", "m3gnet.json", "m3gnet.index", "m3gnet.data-00000-of-00001"],
}

GITHUB_RAW_LINK = "https://raw.githubusercontent.com/materialsvirtuallab/m3gnet/main/pretrained/{model_name}/{filename}"


def _download_file(url: str, target: str):
    logger.info(f"Downloading {target} from {url} ... ")
    if not os.path.isfile(target):
        urllib.request.urlretrieve(url, target)


def _download_model_to_dir(model_name: str = "MP-2021.2.8-EFS", dirname: str = "MP-2021.2.8-EFS"):
    if model_name not in MODEL_FILES:
        raise ValueError(f"{model_name} not supported. Currently we only have {MODEL_FILES.keys()}")
    full_dirname = os.path.join(CWD, dirname)
    if not os.path.isdir(full_dirname):
        os.mkdir(full_dirname)
    for filename in MODEL_FILES[model_name]:
        _download_file(
            GITHUB_RAW_LINK.format(model_name=model_name, filename=filename), os.path.join(full_dirname, filename)
        )
    logger.info(f"Model {model_name} downloaded to {full_dirname}")


@register_plain
class M3GNet(GraphModelMixin, tf.keras.models.Model):
    """
    The main M3GNet model
    """

    def __init__(
        self,
        max_n: int = 3,
        max_l: int = 3,
        n_blocks: int = 3,
        units: int = 64,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        n_atom_types: int = 94,
        include_states: bool = False,
        readout: str = "weighted_atom",
        task_type: str = "regression",
        is_intensive: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        element_refs: Optional[np.ndarray] = None,
        **kwargs,
    ):
        r"""
        Args:
            max_n (int): number of radial basis expansion
            max_l (int): number of angular expansion
            n_blocks (int): number of convolution blocks
            units (int): number of neurons in each MLP layer
            cutoff (float): cutoff radius of the graph
            threebody_cutoff (float): cutoff radius for 3 body interaction
            n_atom_types (int): number of atom types
            include_states (bool): whether to include states calculation
            readout (str): the readout function type. choose from `set2set`,
                `weighted_atom` and `reduce_atom`, default to `weighted_atom`
            task_type (str): `classification` or `regression`, default to
                `regression`
            is_intensive (bool): whether the prediction is intensive
            mean (float): optional `mean` value of the target
            std (float): optional `std` of the target
            element_refs (np.ndarray): element reference values for each
                element
            **kwargs:
        """
        super().__init__(**kwargs)
        self.graph_converter = RadiusCutoffGraphConverter(cutoff=cutoff, threebody_cutoff=threebody_cutoff)

        if include_states:
            self.graph_converter.set_default_states(np.array([[0.0, 0.0]], dtype="float32"))

        if task_type.lower() == "classification":
            act_final = "sigmoid"
        else:
            act_final = None

        self.featurizer = GraphFeaturizer(
            n_atom_types=n_atom_types,
            atom_embedding_dim=units,
            rbf_type="SphericalBessel",
            max_n=max_n,
            max_l=max_l,
            cutoff=cutoff,
            smooth=True,
        )

        self.feature_adjust = GraphUpdateFunc(MLP([units], activations=["swish"], use_bias=False), "bonds")

        self.basis_expansion = SphericalBesselWithHarmonics(max_n=max_n, max_l=max_l, cutoff=cutoff, use_phi=False)
        update_size = max_n * max_l

        self.three_interactions = [
            ThreeDInteraction(
                update_network=MLP([update_size], activations=["sigmoid"]),
                update_network2=GatedMLP([units], activations=["swish"], use_bias=False),
            )
            for _ in range(n_blocks)
        ]

        self.graph_layers = []

        # n_blockは3, やりすぎるとなんかよくないという旨がmegnet論文に書いてあった気がする。
        for i in range(n_blocks):
            atom_network = GatedAtomUpdate(neurons=[units, units], activation="swish")

            bond_network = ConcatAtoms(neurons=[units, units], activation="swish")

            if include_states:
                atom_agg_func = AtomReduceState()
                state_network = ConcatBondAtomState(
                    update_func=MLP([units, units], activations=["swish", "swish"]),
                    atom_agg_func=atom_agg_func,
                    bond_agg_func=None,
                )
            else:
                state_network = StateNetwork()

            layer = GraphNetworkLayer(
                atom_network=atom_network,
                bond_network=bond_network,
                state_network=state_network,
            )
            self.graph_layers.append(layer)

            ## ここまでで、グラフコンボリューションが終了している。あとは出力情報に応じて全結合層を組んでいる。と思う。

        # トレースのための確認
        # print('★★')
        # print(is_intensive)
        # print(readout)
        print(f"units:{units}")
        if is_intensive:
            if readout == "set2set":
                atom_readout = Set2Set(units=units, num_steps=2, field="atoms")

            elif readout == "weighted_atom":
                atom_readout = WeightedReadout(neurons=[units, units], field="atoms")
            else:
                atom_readout = ReduceReadOut("mean", field="atoms")

            readout_nn = MultiFieldReadout(atom_readout=atom_readout, include_states=include_states)

            mlp = MLP([units, units, 1], ["swish", "swish", act_final], is_output=True)

            self.final = Pipe(layers=[readout_nn, mlp])
        
        # エネルギーはこっち
        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive")
            final_layers = []
            print(f"unit:{units}")
            if include_states:
                final_layers.append(
                    GraphNetworkLayer(atom_network=GatedAtomUpdate(neurons=[units], activation="swish"))
                )

            final_layers.append(
                GraphNetworkLayer(
                    atom_network=GraphUpdateFunc(
                        update_func=GatedMLP(
                            neurons=[units, units, 1],
                            activations=["swish", "swish", None],
                        ),
                        update_field="atoms",
                    )
                )
            )
            # このfinal_layersのReduceReadOut層でエネルギーの合計値を計算している。
            # ということはこの一つ前で、各原子のエネルギーが出ている。
            final_layers.append(ReduceReadOut(method="sum", field="atoms"))
            self.final = Pipe(layers=final_layers)

        if element_refs is None:
            self.element_ref_calc = BaseAtomRef()
        else:
            self.element_ref_calc = AtomRef(property_per_element=element_refs, max_z=n_atom_types)
        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = n_blocks
        self.n_atom_types = n_atom_types
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_states = include_states
        self.readout = readout
        self.task_type = task_type
        self.is_intensive = is_intensive
        self.kwargs = kwargs
        self.mean = mean
        self.std = std
        self.element_refs = element_refs

    def call(self, graph: List, **kwargs) -> tf.Tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns:

        """

        # GNNで情報を変換する前にデータを出力する。
        # print('★★')
        # print(len(graph))
        # print(f"atoms:{graph[0]}")
        # print(graph[1])
        # print(graph[2])
        # print(f"posi:{graph[3]}")
        # print('★★')


        graph = tf_compute_distance_angle(graph)
        property_offset, ref_energy_array = self.element_ref_calc(graph)
        three_basis = self.basis_expansion(graph)
        three_cutoff = polynomial(graph[Index.BONDS], self.threebody_cutoff)
        g = self.featurizer(graph)
        g = self.feature_adjust(g)
        for i in range(self.n_blocks):
            print(f"block{i+1}回目")
            g = self.three_interactions[i](g, three_basis, three_cutoff)
            g = self.graph_layers[i](g)

        
        ## ここで各原子の特徴ベクトルを取得する。
        ## 
        atoms_vector = g[Index.ATOMS]
        print("gのATOMS情報")
        print(g[Index.ATOMS][0])
        print("graphのATOMS情報。これは徐の徐の原子番号しか格納されていない。")
        print(graph[Index.ATOMS])



        print('ここからエネルギー------')
        # print(graph[Index.ATOM_POSITIONS])
        # print(graph[Index.ATOMS])
        # print('------')
        g, diff_energy_array = self.final(g)
        print(graph[Index.N_ATOMS][0])
        print(f'std:{self.std} type:{type(self.std)}')
        # 標準化した状態から元の状態に戻してる。そもそもGNNを通って出てきたdiff_Eが標準化されているという仮定はどこから来たのだろう。謎。
        g = g * self.std + self.mean
        # print(f'1個前のg:{g}')
        g += property_offset
        # print(property_offset)
        # print(f'g:{g}, type{type(g)}')
        # print(g[0])

        # 原子エネルギーをnp.arrayで算出
        energy_array = ref_energy_array + diff_energy_array*self.std + self.mean
        print(energy_array)
        print(np.sum(energy_array))

        # dataframe作成
        # self.unitsがatom_embedding_dimなので。
        columns = ['atomic_number', 'x', 'y', 'z', 'energy']
        atom_vector = [f"atom_feature_vector_{i+1}" for i in range(self.units)]
        columns.extend(atom_vector)
        print(columns)
        print(energy_array)

        print(atoms_vector)
        print(self.units)

        d = np.concatenate([graph[Index.ATOMS],graph[Index.ATOM_POSITIONS], energy_array, atoms_vector], axis=1)
        df = pd.DataFrame(data=d, columns=columns)
        # print(f'each_atoms_energy:{field}')
        # print(f'n_field:{n_field}')
        # print(get_segment_indices_from_n(n_field))
        path = os.getcwd()
        df.to_csv(f'{path}/each_atom_energy.csv')

        # dataframe作成
        # columns = [,'mean', 'std']
        # d = np.concatenate([self.mean]*graph[Index.N_ATOMS][0].numpy(), [self.std]*graph[Index.N_ATOMS][0].numpy(), axis=1)
        # df = pd.DataFrame(data=np.array([self.mean,self.std]*graph[Index.N_ATOMS][0].numpy()).reshape(-1,2), columns=columns)
        # print(f'each_atoms_energy:{field}')
        # print(f'n_field:{n_field}')
        # print(get_segment_indices_from_n(n_field))
        # path = os.getcwd()
        # df.to_csv(f'{path}/each_atom_mean_std.csv')


        return g

    def get_config(self):
        """
        Get config dict for serialization
        Returns:
        """
        config = {"name": self.name}
        config.update(
            {
                "max_n": self.max_n,
                "max_l": self.max_l,
                "n_blocks": self.n_blocks,
                "units": self.units,
                "cutoff": self.cutoff,
                "threebody_cutoff": self.threebody_cutoff,
                "include_states": self.include_states,
                "readout": self.readout,
                "n_atom_types": self.n_atom_types,
                "task_type": self.task_type,
                "is_intensive": self.is_intensive,
                "mean": self.mean,
                "std": self.std,
                "element_refs": self.element_refs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "M3GNet":
        r"""
        Construct the model from a config dict
        Args:
            config (dict): config dict from `get_config` method
        Returns: new M3GNet instance
        """
        return cls(**config)

    def save(self, dirname: str):
        """
        Saves the model to a directory.

        Args:
            dirname (str): directory to save the model
        """
        model_serialized = self.to_json()
        model_name = os.path.join(dirname, MODEL_NAME)
        self.save_weights(model_name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fname = os.path.join(dirname, MODEL_NAME + ".json")
        with open(fname, "w") as f:
            json.dump(model_serialized, f)

    @classmethod
    def from_dir(cls, dirname: str, custom_objects: Optional[dict] = None) -> "M3GNet":
        """
        Load the model from a directory

        Args:
            dirname (str): directory to save the model
            custom_objects (dict): dictionary for custom object
        Returns: M3GNet model
        """
        custom_objects = custom_objects or {}
        model_name = os.path.join(dirname, MODEL_NAME)
        fname = os.path.join(dirname, MODEL_NAME + ".json")
        if not os.path.isfile(fname):
            raise ValueError("Model does not exists")
        with open(fname) as f:
            model_serialized = json.load(f)
        # model_serialized = _replace_compatibility(model_serialized)
        model = tf.keras.models.model_from_json(model_serialized, custom_objects=custom_objects)
        model.load_weights(model_name)
        return model

    def set_element_refs(self, element_refs: np.ndarray):
        """
        Set element reference for the property
        Args:
            element_refs (np.ndarray): element reference value for the
                extensive property
        """
        self.element_refs = element_refs
        self.element_ref_calc = AtomRef(property_per_element=element_refs)

    @classmethod
    def load(cls, model_name: str = "MP-2021.2.8-EFS") -> "M3GNet":
        """
        Load the model weights from pre-trained model
        Args:
            model_name (str): model name or the path for saved model. Defaults to "MP-2021.2.8-EFS".

        Returns: M3GNet object.
        """
        if model_name in MODEL_PATHS:
            try:
                return cls.from_dir(MODEL_PATHS[model_name])
            except ValueError:
                _download_model_to_dir(model_name, model_name)
                return cls.load(os.path.join(CWD, model_name))

        if os.path.isdir(model_name):
            if "m3gnet.json" in os.listdir(model_name):
                return cls.from_dir(model_name)

        raise ValueError(f"{model_name} not found in available pretrained {list(MODEL_PATHS.keys())}")
