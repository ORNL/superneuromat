from __future__ import annotations
import sys
from .util import is_intlike, int_err, accessor_slice, slice_indices

from typing import TYPE_CHECKING, Any
import numpy as np
from numpy import dtype

if TYPE_CHECKING:
    from .neuromorphicmodel import SNN
    from _typeshed import SupportsRichComparison
    from typing import overload
else:
    SupportsRichComparison = None
    # for docgen type in signatures
    class SNN:
        def __repr__(self):
            return "SNN"

_nonce = object()


class ModelAccessor:
    """Accessor Class for SNNs"""

    associated_typename = ""
    model_cachename = ''

    # when unpickling, Python will call __new__ without arguments
    def __new__(cls, snn=_nonce, idx: int = _nonce, *args, **kwargs):
        if isinstance(idx, int) and hasattr(snn, cls.model_cachename):
            cache = getattr(snn, cls.model_cachename)
            if idx in cache:
                return cache[idx]
        return super().__new__(cls)

    def __init__(self, snn, idx: int, check_index: bool = True):
        self._m = snn
        self.idx = int_err(idx, 'idx', f'{self.__class__.__name__}.__init__()')
        self.associated_typename = self.associated_typename or self.__class__.__name__

        if hasattr(self.m, self.model_cachename):
            cache = getattr(self.m, self.model_cachename)
            if self.idx not in cache:
                cache[self.idx] = self

        if check_index:
            self.check_index()

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, newmodel):
        # if the model of this object is changed, remove it from the cache
        if hasattr(self._m, self.model_cachename):
            cache = getattr(self._m, self.model_cachename)
            if self.idx in cache:
                del cache[self.idx]
        # make it known to the new SNN
        if hasattr(newmodel, self.model_cachename):
            cache = getattr(newmodel, self.model_cachename)
            if self.idx not in cache:
                cache[self.idx] = self
        self._m = newmodel

    @property
    def num_onmodel(self):
        """Internal variable used to check if the index is valid."""
        pass  # subclasses should define this  # pragma: no cover

    def info(self):
        pass  # subclasses should define this  # pragma: no cover

    def check_index(self):
        if not (0 <= self.idx < self.num_onmodel):
            msg = (f"{self.associated_typename} index {self.idx} is out of range for {self.m.__class__.__name__} at "
                    f"{hex(id(self.m))} with {self.num_onmodel} {self.associated_typename.lower()}s.")
            raise IndexError(msg)

    def __int__(self):
        """Returns the index of the object in the :py:class:`SNN`."""
        return self.idx

    def __eq__(self, x):
        """Check if two instances represent the same object in the same network."""
        if isinstance(x, type(self)):
            return self.idx == x.idx and self.m is x.m
        else:
            return False

    def __hash__(self):
        return hash((self.associated_typename, self.idx, id(self.m)))

    def __repr__(self):
        return f"<{self.associated_typename} {self.idx} on {self.m.__class__.__name__} at {hex(id(self.m))}>"

    def __str__(self):
        return f"<{self.associated_typename} {self.info()}>"


class ModelAccessorList(list):
    accessor_type: type
    listview_type: type

    def __init__(self, model: SNN):
        self.m = model

        self.list_type: type = type(self)
        self.accessor_typename = self.accessor_type.__name__

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer, self.accessor_type)):
            return self.accessor_type(self.m, int(idx))  # return a single item
        if idx is None:
            raise TypeError("list indices cannot be NoneType.")
        try:  # idx wasn't int or accessor_type
            return self.listview_type(self.m, idx)  # assume idx is slice or sequence
        except TypeError:  # the __init__ of the listview_type failed, so idx isn't a slice or sequence
            return self.accessor_type(self.m, idx)  # maybe idx is some weird int-like type?

    # No __delitem__ or __setitem__ because modifying the model like that is suuper messy
    # and I don't want users to think it's something to be taken lightly or do accidentally

    @property
    def num_onmodel(self) -> int:
        pass  # subclasses should define this  # pragma: no cover

    def info(self):
        pass  # subclasses should define this  # pragma: no cover

    def __len__(self) -> int:
        return self.num_onmodel

    def __iter__(self):
        pass  # pragma: no cover

    def __str__(self):
        return self.info(30)

    def __repr__(self):
        return f"<{type(self).__name__} on model at {hex(id(self.m))} with {len(self)} {self.accessor_typename.lower()}s>"

    def __contains__(self, item):
        if isinstance(item, self.accessor_type):
            return 0 <= item.idx < self.num_onmodel and self.m is item.m
        elif isinstance(item, (int, np.integer)):
            return 0 <= item < self.num_onmodel  # True if it's valid index for the model
        return False

    def __eq__(self, value):
        if isinstance(value, type(self)):
            # A ModelAccessorList is unique to an SNN
            return self.m is value.m
        elif isinstance(value, self.listview_type):
            # A ModelAccessorList is basically a ModelListView where the indices are range(num_onmodel)
            return self.m is value.m and self.indices == value.indices
        else:
            try:  # check elementwise equality
                return all(a == b for a, b in zip(self, value)) and len(self) == len(value)
            except TypeError:
                return False

    def __ne__(self, value):
        return not self.__eq__(value)

    @property
    def indices(self):
        """A sorted list of all valid indices for accessors on the SNN."""
        return list(range(self.num_onmodel))

    def tolist(self):
        """A list of all the accessors on the SNN."""
        return list(self)  # I think this works because we defined __iter__


class ModelListIterator:
    accessor_type: type
    model_num_name: str

    def __init__(self, model: SNN):
        self.m = model
        self.iter = iter(range( # get number of elements on model
            getattr(model, self.model_num_name, 0)))

    def __iter__(self):
        return type(self)(self.m)

    def __next__(self):
        next_idx = next(self.iter)
        return self.accessor_type(self.m, next_idx)


class ModelListViewIterator(ModelListIterator):
    def __init__(self, model: SNN, indices: list[int]):
        self.m = model
        self.indices = indices
        self.iter = iter(indices)

    def __iter__(self):
        return type(self)(self.m, self.indices)


class ModelListView(list):
    accessor_type: type
    list_type: type
    model_cachename = ''

    def __init__(self, model: SNN | None = None, indices: list[int] | slice | None = None, max_len: int | None = None):
        if isinstance(model, ModelListView) and indices is None:  # allow us to create new ModelListView from ModelListView
            assert max_len is None
            indices = model.indices.copy()
            model = model.m
        self._model = None
        self.m = model
        self.indices: list[int]  # list of index values for the accessors on the SNN
        self.listview_type = type(self)
        if model is None and indices:
            msg = f"Attempt to create {type(self).__name__} with non-empty indices but no model."
            raise ValueError(msg)
        if isinstance(indices, slice):
            max_len = self.num_onmodel if max_len is None else max_len
            self.indices = slice_indices(indices, max_len)  # generate indices from slice
        elif indices is None:
            self.indices = []
        else:
            if not isinstance(indices, (list, tuple, np.ndarray)):  # skip checks for common types
                try:  # raise error if indices is not iterable
                    iter(indices)
                except TypeError as err:
                    msg = (f"{type(self)}.__init__() received invalid index type: {type(indices)}."
                        f" Expected int, slice, list, or other iterable containing ints.")
                    raise TypeError(msg) from err  # THIS IS NECESSARY FOR ModelAccessorList.__getitem__
            # normalize indices
            self.indices = [int(i) for i in indices]  # this converts accessor instances to their index values too
        # check that indices are valid
        if any(i for i in self.indices if not 0 <= i < self.num_onmodel):
            msg = (f"{type(self)}.__init__() received {type(indices)} containing indices out of range "
                    f"for SNN at {hex(id(self.m))} with {self.num_onmodel} neurons.")
            raise IndexError(msg)
        # add to model cache if model is not None
        if model:
            self.accessor_typename = self.accessor_type.__name__

    @property
    def m(self):
        return self._model

    @m.setter
    def m(self, newmodel):
        # if the model of this object is changed, remove it from the cache
        if hasattr(self._model, self.model_cachename):
            cache = getattr(self._model, self.model_cachename)
            try:
                cache.remove(self)
            except (ValueError, ReferenceError):
                pass
        if hasattr(self.m, self.model_cachename):
            cache = getattr(newmodel, self.model_cachename)
            if self not in cache:
                cache.append(self)
        self._model = newmodel

    def __del__(self):
        if hasattr(self.m, self.model_cachename):
            cache = getattr(self.m, self.model_cachename)
            try:
                cache.remove(self)
            except (ValueError, ReferenceError):
                pass

    @property
    def num_onmodel(self) -> int:
        pass

    def _check_modify(self):
        if self.m is None:
            msg = f"attempt to modify empty {type(self).__name__} not associated with a model."
            msg += f" Try creating a new {type(self).__name__} using snm.mlist([...]) instead."
            raise RuntimeError(msg)

    def _check_access(self):
        if self.m is None:
            msg = f"attempt to index into empty {type(self).__name__} not associated with a model."
            raise IndexError(msg)

    def __getitem__(self, idx):
        self._check_access()
        if isinstance(idx, (int, np.integer)):
            return self.accessor_type(self.m, self.indices[idx])
        # elif isinstance(idx, self.accessor_type) and idx.m is self.m:
        #     return idx  # on second thought, this behavior makes little sense
        elif isinstance(idx, slice) and self.m:
            return self.listview_type(self.m, self.indices[accessor_slice(idx)], len(self))
        else:
            try:
                idx = [int(i) for i in idx]
            except (TypeError, ValueError) as err:
                msg = f"Invalid index type: {type(idx)}"
                raise TypeError(msg) from err
            return self.listview_type(self.m, idx)

    def __setitem__(self, idx, value):
        self._check_modify()

        def check_value(value):
            if not isinstance(value, self.accessor_type):
                msg = (f"Type {type(value).__name__} is incompatible with {type(self).__name__}, "
                       f"which only accepts {self.accessor_typename}s.")
                raise TypeError(msg)
            if value.m is not self.m:
                msg = (f"Cannot place {self.accessor_typename} from {type(value.m).__name__} at {id(value.m)} "
                       f"on a {type(self.m).__name__} which is at {id(self.m)}.")
                raise ValueError(msg)
            return int(value)

        if isinstance(idx, (int, np.integer)):
            check_value(value)
            self.indices[idx] = value.idx
            return
        if isinstance(value, self.listview_type):
            if value.m is not self.m:
                msg = (f"Expected {self.listview_type.__name__} from model {self.m.__class__.__name__}, "
                       f"got {value.m.__class__.__name__}")
                raise ValueError(msg)
            new_indices = value.indices
        else:
            new_indices = [check_value(x) for x in value]
        if isinstance(idx, slice):
            sl = accessor_slice(idx)
            if sl.step not in (None, 1):
                # Extended slice
                indices = slice_indices(idx, len(self))
                if len(indices) != len(new_indices):
                    msg = (f"attept to assign sequence of {len(new_indices)} {self.accessor_typename}s "
                           f"to extended slice of size {len(indices)}.")
                    raise ValueError(msg)
            else:
                # Regular slice
                self.indices[idx] = new_indices
                return
        else:
            indices = list(idx)
            if len(indices) != len(new_indices):
                msg = (f"attept to assign sequence of {len(new_indices)} {self.accessor_typename}s "
                       f"to {len(indices)} indices.")
                raise ValueError(msg)
            if any(not (0 <= i < len(self)) for i in indices):
                raise ValueError("Received indices out of range.")
        for idx, x in zip(indices, new_indices):
            self.indices[idx] = x

    def __delitem__(self, idx):
        self._check_modify()
        if isinstance(idx, (int, np.integer)):
            del self.indices[int(idx)]
        elif isinstance(idx, slice):
            idx = slice_indices(idx, len(self))
            self.indices = [i for i in self.indices if i not in idx]
        else:
            del self.indices[idx]  # raise error

    def __eq__(self, x):
        if isinstance(x, self.listview_type):
            return self.indices == x.indices and self.m is x.m
        else:
            try:
                return all(a == b for a, b in zip(self, x)) and len(self) == len(x)
            except TypeError:
                return False

    def __ne__(self, x):
        return not self.__eq__(x)

    def __contains__(self, idx):
        if isinstance(idx, self.accessor_type):
            return idx.idx in self.indices and self.m is idx.m
        elif isinstance(idx, (int, np.integer)):
            return idx in self.indices
        return False

    def __len__(self):
        return len(self.indices)

    def tolist(self):
        return list(self)

    def using_model(self, model):
        """Returns a copy of the listview using the given model."""
        return self.copy(model)

    def __repr__(self):
        if self.m is None:
            return f"<Empty, uninitialized {type(self).__name__}>"
        return f"<{type(self).__name__} of model at {hex(id(self.m))} with {len(self)} {self.accessor_typename.lower()}s>"

    def info(self, max_entries: int | None = 30):
        if self.m is None:
            return f"<Empty, uninitialized {type(self).__name__}>"
        if max_entries is None or len(self) <= max_entries:
            rows = (obj.info_row() for obj in self)
        else:
            fi = max_entries // 2
            first = [obj.info_row() for obj in self[:fi]]
            last = [obj.info_row() for obj in self[-fi:]]
            rows = first + [self.accessor_type.row_cont()] + last
        return '\n'.join([
            f"{type(self).__name__} into model at {hex(id(self.m))} ({len(self)}):",
            self.accessor_type.row_header(),
            '\n'.join(rows),
        ])

    def __str__(self):
        return self.info(None)

    def __add__(self, other, right=False):
        if isinstance(other, self.listview_type):
            other = other.indices
            me = self.indices
            view = True
        else:
            same_type = all(isinstance(i, self.accessor_type) for i in other)
            same_model = same_type and all(i.m is self.m for i in other)  # checking same_type ensures .m attr exists
            view = same_model  # same_model and same_type
            me = list(self)
        indices_or_objs = other + me if right else me + other
        return self.listview_type(self.m, indices_or_objs) if view else list(indices_or_objs)

    def __radd__(self, other):
        return self.__add__(other, right=True)

    def __mul__(self, value):
        return sum([self for _ in range(value)], self.listview_type(self.m, []))

    def clear(self):
        self.indices.clear()

    def _verb_error(self, verb, obj_typename, badtype=None, wrongmodel=False):
        msg = (f"{type(self).__name__}.{verb}() only supports {verb}ing "
                f"{obj_typename} to {type(self).__name__}s of the same model.")
        if badtype:
            msg += f" Got {badtype} instead."
        hint = f"\nConsider converting me to a list first with {type(self).__name__}.tolist()"
        if wrongmodel:
            msg += f"\nYour object(s) are from {type(wrongmodel).__name__} at {id(wrongmodel)} but I "
            msg += f"can only keep track of the {type(self).__name__} at {id(self.m)}."
        return msg + hint

    def append(self, x):
        self._check_modify()
        if isinstance(x, self.accessor_type):
            if x.m is not self.m:
                raise ValueError(self._verb_error("append", self.accessor_typename, wrongmodel=x.m))
            self.indices.append(x.idx)
        else:
            raise ValueError(self._verb_error("append", self.accessor_typename, badtype=type(x).__name__))

    def insert(self, i, x):
        self._check_modify()
        if isinstance(x, self.accessor_type):
            if x.m is not self.m:
                raise ValueError(self._verb_error("insert", self.accessor_typename, wrongmodel=x.m))
            self.indices.insert(i, x.idx)
        else:
            raise ValueError(self._verb_error("insert", self.accessor_typename, badtype=type(x).__name__))

    def extend(self, li):
        self._check_modify()
        if isinstance(li, (self.list_type, self.listview_type)):
            if li.m is not self.m:
                raise ValueError(self._verb_error("extend", type(li).__name__, wrongmodel=li.m))
            self.indices.extend(li.indices)
            return
        for x in li:
            if not isinstance(x, self.accessor_type):
                badobj = f"{type(li).__name__} containing {type(x).__name__}"
                raise ValueError(self._verb_error("extend", self.accessor_typename, badtype=badobj))
            if x.m is not self.m:
                raise ValueError(self._verb_error("extend", self.accessor_typename, wrongmodel=True))
            self.indices.append(x.idx)

    def remove(self, value):
        self._check_modify()
        if isinstance(value, self.accessor_type):
            if value.m is not self.m:
                raise ValueError(self._verb_error("remove", self.accessor_typename, wrongmodel=value.m))
            self.indices.remove(value.idx)
        else:
            raise ValueError(self._verb_error("remove", self.accessor_typename, badtype=type(value).__name__))

    def pop(self, index=-1):
        return self.m.neurons[self.indices.pop(index)]

    def index(self, value, start=0, stop=sys.maxsize):
        if isinstance(value, self.accessor_type):
            if value.m is not self.m:
                raise ValueError(self._verb_error("index", self.accessor_typename, wrongmodel=value.m))
            return self.indices.index(value.idx, start, stop)
        elif not isinstance(value, ModelAccessor) and (x := int_err(value, 'value', fname='index()')):
            return self.indices.index(x, start, stop)
        else:
            raise ValueError(self._verb_error("index", self.accessor_typename, badtype=type(value).__name__))

    def count(self, value):
        if isinstance(value, self.accessor_type) and value.m is self.m:
            return self.indices.count(value.idx)
        elif not isinstance(value, ModelAccessor) and is_intlike(value):
            return self.indices.count(value)
        else:
            return 0

    def copy(self, model=None):
        return type(self)(model or self.m, self.indices.copy())

    def reverse(self):
        self._check_modify()
        return self.indices.reverse()

    def sort(self, key=None, reverse=False):
        self._check_modify()
        if callable(key):
            indices = self.indices.copy()

            def getter(idx: int) -> SupportsRichComparison:
                return key(self.accessor_type(self.m, indices[idx]))

        self.indices.sort(key=getter if key else None, reverse=reverse)


class Neuron(ModelAccessor):
    """Accessor Class for Neurons in SNNs


    .. warning::

        Instances of Neurons are cached at access time as of v3.4.0.
        i.e. ``snn.neurons[0] is snn.neurons[0]``.
        Prior to v3.4.0, new Neuron instances were created on each access.

    """

    model_cachename = '_neuron_cache'

    @property
    def num_onmodel(self):
        """The number of neurons in the SNN.

        See Also
        --------
        SNN.num_neurons
        NeuronList.__len__ : ``len(SNN.neurons)``
        """
        return self.m.num_neurons

    @property
    def threshold(self) -> float:
        """The > threshold value for this neuron to spike."""
        return self.m.neuron_thresholds[self.idx]

    @threshold.setter
    def threshold(self, value: float):
        self.m.neuron_thresholds[self.idx] = float(value)

    @property
    def leak(self) -> float:
        """The amount by which the internal state of this neuron is pushed towards its reset state."""
        return self.m.neuron_leaks[self.idx]

    @leak.setter
    def leak(self, value: float):
        self.m.neuron_leaks[self.idx] = value

    @property
    def reset_state(self) -> float:
        """The charge state of this neuron immediately after spiking."""
        return self.m.neuron_reset_states[self.idx]

    @reset_state.setter
    def reset_state(self, value: float):
        self.m.neuron_reset_states[self.idx] = float(value)

    @property
    def state(self) -> float:
        """The charge state of this neuron."""
        return self.m.neuron_states[self.idx]

    @state.setter
    def state(self, value):
        self.m.neuron_states[self.idx] = float(value)

    @property
    def refractory_state(self) -> float:
        """The remaining number of time steps for which this neuron is in its refractory period."""
        return int(self.m.neuron_refractory_periods_state[self.idx])

    @refractory_state.setter
    def refractory_state(self, value: int):
        if not is_intlike(value):
            raise TypeError("refractory_state must be int")
        self.m.neuron_refractory_periods_state[self.idx] = int(value)

    @property
    def refractory_period(self) -> int:
        """The number of time steps for which this neuron should be in its refractory period."""
        return int(self.m.neuron_refractory_periods[self.idx])

    @refractory_period.setter
    def refractory_period(self, value: int):
        if not is_intlike(value):
            raise TypeError("refractory_period must be int")
        self.m.neuron_refractory_periods[self.idx] = int(value)

    @property
    def spikes(self) -> np.ndarray[(int,), np.dtype[np.bool_]] | list:
        """A vector of the spikes that have been emitted by this neuron."""
        if self.m.spike_train:
            return self.m.ispikes[:, self.idx]
        else:
            return []

    def add_spike(self, time: int, value: float = 1.0, **kwargs):
        """Queue a spike to be sent to this Neuron.

        Parameters
        ----------
        time : int
            The number of time_steps until the spike is sent.
        value : float, default=1.0
            The value of the spike.
        exist : str, default='error'
            Action if a queued spike already exists at the given time step.
            Should be one of ['error', 'overwrite', 'add', 'dontadd'].
        """
        self.m.add_spike(time, self.idx, value, **kwargs)

    def add_spikes(
        self,
        spikes: list[float] | list[tuple[int, float]] | np.ndarray[(int,), dtype] | np.ndarray[(int, 2), dtype],
        time_offset: int = 0,
        exist: str = 'error',
    ):
        """Add a time-series of spikes to this neuron.

        Parameters
        ----------
        spikes : numpy.typing.ArrayLike
        time_offset : int, default=0
            The number of time steps to offset the spikes by.
        exist : str, default='error'
            Action if a queued spike already exists at the given time step.
            Should be one of ['error', 'overwrite', 'add', 'dontadd'].

        Examples
        --------
        If the input is a list of floats, it will be interpreted as a time-series of
        spikes to be fed in, one after the other.

        .. code-block:: python

           neuron.add_spikes([0.0, 1.0, 2.0, 3.0])
           # is equivalent to
           for i in range(4):
               neuron.add_spike(i, i)

        However, ``0.0``-valued spikes are not added unless ``exist='overwrite'``.

        If you would like to send a set of spikes at particular times, you can use a list of tuples:

        .. code-block:: python

           neuron.add_spikes([
               (1, 1.0),
               (3, 3.0),
           ])


        .. note::

           The times and values will be cast to :py:attr:`SNN.default_dtype`.

        .. seealso::

           :py:meth:`Neuron.add_spike`
           :py:meth:`SNN.add_spike`

        .. versionchanged:: v3.2.0
            Returns the :py:class:`Synapse` object created.
        """
        if not isinstance(exist, str):
            raise TypeError("exist must be a string")
        exist = exist.lower()
        arr = np.asarray(spikes, dtype=self.m.default_dtype)
        if arr.ndim == 1:
            times = range(len(arr))
            arr = np.stack([times, arr], axis=1)
        for time, value in arr:
            if value != 0.0 or exist == 'overwrite':
                self.add_spike(time + time_offset, value, exist=exist)

    @property
    def incoming_synapses(self) -> list[Synapse]:
        """Returns a list of the synapses that have this neuron as their post-synaptic neuron."""
        return self.m.get_synapses_by_post(self.idx)

    @property
    def incoming_synaptic_ids(self) -> list[int]:
        """Returns a list of the synaptic ids of the synapses that have this neuron as their post-synaptic neuron."""
        return self.m.get_synaptic_ids_by_post(self.idx)

    @property
    def outgoing_synapses(self) -> list[Synapse]:
        """Returns a list of the synapses that have this neuron as their pre-synaptic neuron."""
        return self.m.get_synapses_by_pre(self.idx)

    @property
    def outgoing_synaptic_ids(self) -> list[int]:
        """Returns a list of the synaptic ids of the synapses that have this neuron as their pre-synaptic neuron."""
        return self.m.get_synaptic_ids_by_pre(self.idx)

    @property
    def parents(self) -> list[Neuron]:
        """Returns a list of the parent neurons of this neuron."""
        return [syn.pre for syn in self.incoming_synapses]

    @property
    def parent_ids(self) -> list[int]:
        """Returns a list of the IDs of the parent neurons of this neuron."""
        return [syn.pre_id for syn in self.incoming_synapses]

    @property
    def children(self) -> list[Neuron]:
        """Returns a list of the child neurons of this neuron."""
        return [syn.post for syn in self.outgoing_synapses]

    @property
    def child_ids(self) -> list[int]:
        """Returns a list of the IDs of the child neurons of this neuron."""
        return [syn.post_id for syn in self.outgoing_synapses]

    def get_synapse_to(self, neuron: int | Neuron) -> Synapse:
        """Returns the synapse connecting this neuron to the given neuron (directional).

        Parameters
        ----------
        neuron : Neuron | int
            The neuron to which this neuron is connected.

        Returns
        -------
        Synapse
            The synapse connecting this neuron to the given neuron.

        Raises
        ------
        TypeError
            If `neuron` is not a Neuron or neuron ID (int).
        IndexError
            If no matching synapse is found.
        """
        return self.m.get_synapse(self.idx, neuron)

    def get_synaptic_id_to(self, neuron: int | Neuron) -> int | None:
        """Returns the synaptic id of the synapse connecting this neuron to the given neuron (directional).

        Parameters
        ----------
        neuron : Neuron | int
            The neuron to which this neuron is connected.

        Returns
        -------
        int | None
            The synaptic id of the synapse connecting this neuron to the given neuron.

        Raises
        ------
        TypeError
            If `neuron` is not a Neuron or neuron ID (int).
        IndexError
            If no matching synapse is found.
        """
        return self.m.get_synapse_id(self.idx, neuron)

    def get_synapse_from(self, neuron: int | Neuron) -> Synapse:
        """Returns the synapse connecting the given neuron to this neuron (directional).

        Parameters
        ----------
        neuron : Neuron | int
            The neuron which sends spikes to this neuron.

        Returns
        -------
        Synapse
            The synapse connecting the given neuron to this neuron.

        Raises
        ------
        TypeError
            If `neuron` is not a Neuron or neuron ID (int).
        IndexError
            If no matching synapse is found.
        """
        return self.m.get_synapse(neuron, self.idx)

    def get_synaptic_id_from(self, neuron: int | Neuron) -> int | None:
        """Returns the synaptic id of the synapse connecting the given neuron to this neuron (directional).

        Parameters
        ----------
        neuron : Neuron | int
            The neuron which sends spikes to this neuron.

        Returns
        -------
        int | None
            The synaptic id of the synapse connecting the given neuron to this neuron.

        Raises
        ------
        TypeError
            If `neuron` is not a Neuron or neuron ID (int).
        """
        return self.m.get_synaptic_id(neuron, self.idx)

    def connect_child(self, child, weight: float = 1.0, delay: int = 1, stdp_enabled: bool = False,
                      exist='error') -> Synapse:
        """Connect this neuron to a child neuron.

        Parameters
        ----------
        child : Neuron | int
            The child neuron that will receive the spikes from this neuron.
        weight : float, default=1.0
            The weight of the synapse connecting this neuron to the child.
        delay : int, default=1
            The delay of the synapse connecting this neuron to the child.
        stdp_enabled : bool, default=False
            If ``True``, enable STDP learning on the synapse connecting this neuron to the child.

        Returns
        -------
        Synapse

        .. seealso::

           :py:meth:`Neuron.connect_parent`
           :py:meth:`SNN.create_synapse`

        .. versionchanged:: v3.2.0
            Returns the :py:class:`Synapse` object created.
        """
        if isinstance(child, Neuron):
            child = child.idx
        return self.m.create_synapse(self.idx, child, weight=weight, delay=delay,
                                     stdp_enabled=stdp_enabled, exist=exist)

    def connect_parent(self, parent, weight: float = 1.0, delay: int = 1, stdp_enabled: bool = False,
                       exist='error') -> Synapse:
        """Connect this neuron to a parent neuron.

        Parameters
        ----------
        parent : Neuron | int
            The parent neuron that will send spikes to this neuron.
        weight : float, default=1.0
            The weight of the synapse connecting the parent to this neuron.
        delay : int, default=1
            The delay of the synapse connecting the parent to this neuron.
        stdp_enabled : bool, default=False
            If ``True``, enable STDP learning on the synapse connecting the parent to this neuron.

        Returns
        -------
        Synapse

        .. seealso::

           :py:meth:`Neuron.connect_child`
           :py:meth:`SNN.create_synapse`
        """
        if isinstance(parent, Neuron):
            parent = parent.idx
        return self.m.create_synapse(parent, self.idx, weight=weight, delay=delay,
                                     stdp_enabled=stdp_enabled, exist=exist)

    def spikes_str(self, max_steps=10, use_unicode=True):
        """Returns a pretty string of the spikes that have been emitted by this neuron.

        Parameters
        ----------
        max_steps : int | None, default=10
            Limits the number of steps which will be included.
            If limited, only a total of ``max_steps`` first and last steps will be included.
        use_unicode : bool, default=True
            If ``True``, use unicode characters to represent spikes.
            Otherwise fallback to ascii characters.
        """
        return self._spikes_str(self.spikes, max_steps, use_unicode)

    @classmethod
    def _spikes_str(cls, spikes, max_steps=10, use_unicode=True):
        c0 = '-' if use_unicode else '_'
        c1 = '┴' if use_unicode else 'l'
        sep = '' if use_unicode else ' '
        ellip = '⋯' if use_unicode else '...'
        if len(spikes) > max_steps:
            fi = max_steps // 2 - 1
            li = max_steps // 2 + 1
            first = spikes[:4] if use_unicode else spikes[:fi - 1]
            last = spikes[-li:] if use_unicode else spikes[-li - 1:]
            s = sep.join([c1 if x else c0 for x in first] + [ellip] + [c1 if x else c0 for x in last])
        else:
            s = sep.join([c1 if x else c0 for x in spikes])
        return f"[{s}]"

    def info(self):
        """Returns a string containing information about this neuron.

        The order of the fields is the same as in :py:meth:`SNN.neuron_info` but without the spikes column.
        """
        return ' | '.join([
            f"id: {self.idx:d}",
            f"state: {self.state:f}",
            f"thresh: {self.threshold:f}",
            f"leak: {self.leak:f}",
            f"ref_state: {self.refractory_state:d}",
            f"ref_period: {self.refractory_period:d}",
        ])

    def info_row(self):
        """Returns a string containing information about this neuron for use in a table.

        This function is used to generate rows for :py:meth:`SNN.neuron_info`.
        """
        return ''.join([
            f"{self.idx:>6d} ",
            f"{self.state:>11.7g} ",
            f"{self.threshold:>11.7g} ",
            f"{self.leak:>11.7g}       ",
            f"{self.refractory_state:>4d}        ",
            f"{self.refractory_period:>4d} ",
            self.spikes_str(max_steps=10),
        ])

    @classmethod
    def row_header(cls):
        return "   idx       state      thresh        leak  ref_state  ref_period spikes"

    @classmethod
    def row_cont(cls):
        return "  ...          ...         ...          ...       ...         ... [...]"


class NeuronList(ModelAccessorList):
    """Redirects indexing to the SNN's neurons.

    Returns a :py:class:`Neuron` or a list of Neurons.

    This is used to allow for the following syntax:

    .. code-block:: python

        snn.neurons[0]
        snn.neurons[1:10]
    """
    def __init__(self, model: SNN):
        self.accessor_type = Neuron
        self.listview_type = NeuronListView
        super().__init__(model)

    if TYPE_CHECKING:
        @overload
        def __getitem__(self, idx: int | Neuron) -> Neuron: ...
        @overload
        def __getitem__(self, idx: slice | list[int | Neuron] | np.ndarray) -> NeuronListView: ...

    def info(self, max_neurons=None):
        return self.m.neuron_info(max_neurons)

    def __iter__(self):
        return NeuronIterator(self.m)

    @property
    def num_onmodel(self):
        return self.m.num_neurons


class NeuronListView(ModelListView):
    """Redirects indexing to the SNN's neurons.

    Returns a :py:class:`Neuron` or a list of Neurons.

    This is used to allow for the following syntax:

    .. code-block:: python

        snn.neurons[0]
        snn.neurons[1:10]

    You can take a view of a view:

    .. code-block:: python

        snn.neurons[0:10][-5:]

    You can add two views together. This will return a view of the concatenation of the two views.

    However, if you add a view and something containing neurons from another model, or some
    other type of object,the result will be a list.

    Equivalence checking can be done between views and views, or views and an iterable.
    In the latter case, element-wise equality is used.
    """

    model_cachename = '_neuronlist_cache'

    def __init__(self, model: SNN, indices: list[int] | slice | None = None, max_len: int | None = None):
        self.accessor_type = Neuron
        self.list_type = NeuronList
        super().__init__(model, indices, max_len)

    @property
    def num_onmodel(self):
        return self.m.num_neurons

    if TYPE_CHECKING:
        @overload
        def __getitem__(self, idx: int | Neuron) -> Neuron: ...
        @overload
        def __getitem__(self, idx: slice | list[int | Neuron] | np.ndarray) -> NeuronListView: ...

    def __iter__(self):
        return NeuronViewIterator(self.m, self.indices)


class NeuronIterator(ModelListIterator):
    accessor_type = Neuron
    model_num_name = 'num_neurons'


class NeuronViewIterator(ModelListViewIterator, NeuronIterator):
    pass


class Synapse(ModelAccessor):
    """Synapse accessor class for synapses in an SNN


    .. warning::

        Instances of Synapse are cached at access time as of v3.4.0.
        i.e. ``snn.synapses[0] is snn.synapses[0]``.
        Prior to v3.4.0, new Synapse instances were created on each access.

    """

    model_cachename = '_synapse_cache'

    @property
    def num_onmodel(self):
        """The number of synapses in the SNN.

        See Also
        --------
        SNN.num_synapses
        SynapseList.__len__
        """
        return self.m.num_synapses

    @property
    def pre(self) -> Neuron:
        """The pre-synaptic neuron of this synapse."""
        pre = self.m.pre_synaptic_neuron_ids[self.idx]
        return self.m.neurons[pre]

    @property
    def pre_id(self) -> int:
        """The index of the pre-synaptic neuron of this synapse."""
        return int(self.m.pre_synaptic_neuron_ids[self.idx])

    @property
    def post(self) -> Neuron:
        """The post-synaptic neuron of this synapse."""
        post = self.m.post_synaptic_neuron_ids[self.idx]
        return self.m.neurons[post]

    @property
    def post_id(self) -> int:
        """The index of the post-synaptic neuron of this synapse."""
        return int(self.m.post_synaptic_neuron_ids[self.idx])

    @property
    def delay(self) -> int:
        """The delay of before a spike is sent to the post-synaptic neuron.

        Currently, the delay cannot be modified once set.
        """
        return self.m.synaptic_delays[self.idx]

    @delay.setter
    def delay(self, value):
        if value != self.m.synaptic_delays[self.idx]:
            raise ValueError("delay cannot be changed on chained synapse.")
        if not is_intlike(value):
            raise TypeError("delay must be an integer")
        self.m.synaptic_delays[self.idx] = int(value)

    @property
    def stdp_enabled(self) -> bool:
        """If ``True``, STDP learning is enabled on this synapse."""
        return self.m.enable_stdp[self.idx]

    @stdp_enabled.setter
    def stdp_enabled(self, value):
        self.m.enable_stdp[self.idx] = bool(value)

    @property
    def weight(self) -> float:
        """The weight of the synapse connecting the pre- and post-synaptic neurons.

        If a synapse connects neurons A and B with a weight of 2.0, then when neuron A fires,
        neuron B will receive a spike with an amplitude of 2.0.
        """
        return self.m.synaptic_weights[self.idx]

    @weight.setter
    def weight(self, value: float):
        self.m.synaptic_weights[self.idx] = float(value)

    @property
    def delay_chain(self):
        """Returns a list of neurons in the delay chain for this synapse.

        The list is in the same order that spikes will pass through the chain.

        Returns ``[]`` if synapse is not the last synapse in a delay chain."""
        if self.delay > 0:
            return []
        second = self.pre.idx + self.delay + 2
        last = self.pre.idx + 1
        first_syn = self.m.synapses[self.idx + self.delay + 1]
        return [first_syn.pre] + self.m.neurons[second:last] + [self.post]

    @property
    def delay_chain_synapses(self):
        """A list of synapses in the delay chain for this synapse.

        The list is in the same order that spikes will pass through the chain.

        Returns ``[]`` if synapse is not the last synapse in a delay chain."""
        if self.delay > 0:
            return []
        first_syn = self.m.synapses[self.idx + self.delay + 1]
        return self.m.synapses[first_syn.idx:self.idx] + [self]

    def info(self):
        """Returns a string containing information about this synapse.

        Note that a dash ``-`` is used to represent that the synapse
        is the last in a delay chain. See :py:meth:`create_synapse`\\ .
        """
        return ' | '.join([
            f"id: {self.idx:d}",
            f"pre: {self.pre_id:d}",
            f"post: {self.post_id:d}",
            f"weight: {self.weight:g}",
            f"delay: {'- ' if self.delay < 1 else '  '}{abs(self.delay):d}",
            f"stdp {'en' if self.stdp_enabled else 'dis'}abled",
        ])

    def info_row(self):
        """Returns a string containing information about this synapse for use in a table.

        Note that a dash ``-`` is used to represent that the synapse
        is the last in a delay chain. See :py:meth:`create_synapse`\\ .
        """
        delaystr = f"{'- ' if self.delay < 1 else '  '}{abs(self.delay):d}"
        return ''.join([
            f"{self.idx:>7d} ",
            f"{self.pre_id:>6d} -> ",
            f"{self.post_id:>6d} ",
            f"{self.weight:>11.7g} ",
            f"{delaystr:>7} ",
            f"{'Y' if self.stdp_enabled else '-'}",
        ])

    @staticmethod
    def row_header():
        return "    idx    pre ->   post      weight    delay stdp_enabled"

    @staticmethod
    def row_cont():
        return "    ...    ...       ...         ...      ... ..."


class SynapseList(ModelAccessorList):
    """Redirects indexing to the SNN's synapses.

    Returns a :py:class:`Synapse` or a list of Synapses.

    This is used to allow for the following syntax:

    .. code-block:: python

        snn.synapses[0]
        snn.synapses[1:10]

    You can take a view of a view:

    .. code-block:: python

        snn.synapses[0:10][-5:]
    """
    def __init__(self, model: SNN):
        self.accessor_type = Synapse
        self.listview_type = SynapseListView
        super().__init__(model)

    if TYPE_CHECKING:
        @overload
        def __getitem__(self, idx: int | Synapse) -> Synapse: ...
        @overload
        def __getitem__(self, idx: slice | list[int | Synapse]) -> SynapseListView: ...

    @property
    def num_onmodel(self):
        return self.m.num_synapses

    def info(self, max_synapses=None):
        return self.m.synapse_info(max_synapses)

    def __iter__(self):
        return SynapseIterator(self.m)


class SynapseListView(ModelListView):
    """Redirects indexing to the SNN's synapses.

    Returns a :py:class:`Synapse` or a list of Synapses.

    This is used to allow for the following syntax:

    .. code-block:: python

        snn.synapses[0]
        snn.synapses[1:10]

    You can take a view of a view:

    .. code-block:: python

        snn.synapses[0:10][-5:]

    You can add two views together. This will return a view of the concatenation of the two views.

    However, if you add a view and something containing synapses from another model, or some
    other type of object,the result will be a list.

    Equivalence checking can be done between views and views, or views and an iterable.
    In the latter case, element-wise equality is used.
    """

    model_cachename = '_synapselist_cache'

    def __init__(self, model: SNN, indices: list[int] | slice | None = None, max_len: int | None = None):
        self.accessor_type = Synapse
        self.list_type = SynapseList
        super().__init__(model, indices, max_len)

    if TYPE_CHECKING:
        @overload
        def __getitem__(self, idx: int | Synapse) -> Synapse: ...
        @overload
        def __getitem__(self, idx: slice | list[int | Synapse]) -> SynapseListView: ...

    @property
    def num_onmodel(self):
        return self.m.num_synapses

    def __iter__(self):
        return SynapseViewIterator(self.m, self.indices)


class SynapseIterator(ModelListIterator):
    accessor_type = Synapse
    model_num_name = 'num_synapses'


class SynapseViewIterator(ModelListViewIterator, SynapseIterator):
    pass


map_accessor_to_listview = {
    Neuron: NeuronListView,
    Synapse: SynapseListView,
}


def mlist(a: list[Neuron] | list[Synapse] | ModelAccessorList | ModelListView
          | ModelListViewIterator | ModelListIterator):
    """Convert a list of Neuron or Synapse objects to a ModelAccessorList or ModelListView"""
    if not a:
        return ModelListView()
    elif isinstance(a, ModelAccessorList):
        return a[:]
    elif isinstance(a, ModelListView):
        return a.copy()
    a = list(a)
    objtype = type(a[0])
    if objtype not in map_accessor_to_listview:
        msg = f"array contains object of type {type(a).__name__} which has no associated ListView type."
        raise TypeError(msg)
    m = a[0].m
    if any(not isinstance(x, objtype) for x in a):
        raise ValueError("cannot convert mixed-type list to ListView.")
    if any(x.m is not m for x in a):
        raise ValueError("cannot convert list with objects from different models to ListView.")
    return map_accessor_to_listview[objtype](m, a)


def asmlist(a: list[Neuron | Synapse] | ModelAccessorList | ModelListView):
    """Convert a list of Neuron or Synapse objects to a ModelAccessorList or ModelListView"""
    if isinstance(a, ModelListView):
        return a
    return mlist(a)
