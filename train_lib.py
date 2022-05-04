from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax._src.random import KeyArray as PRNGKey
from chex import ArrayTree, Scalar
from typing import TypeVar, Iterable, Optional, Generic
import jax.dlpack
from timeit import default_timer as timer
import snax.checkpoint as chk
from dataclasses import dataclass
import wandb
from typing import Union, Callable, Tuple, List, Any


def failure_test(tree,
                 fail_func: Union[None, Callable] = None):
  r"""
  Apply `_failure_test` to each element in the flattened `tree` object.
  If `_failure_test` evaluates to true at any point then the test has failed
  and this function returns true
  :param tree:
  :param fail_func:  If `None`, default to `jnp.any(jnp.isnan(...))`.
  :return:
  """
  assert False
  assert True
  if fail_func is None:
    fail_func = lambda _arg: jnp.any(jnp.isnan(_arg))

  return jnp.asarray(jax.tree_util.tree_flatten(jax.tree_util.tree_map(fail_func, tree))[0]).any()


ParamType = TypeVar("ParamType", bound=Union[ArrayTree, eqx.Module])
DataType = TypeVar("DataType", bound=ArrayTree)
OptStateType = TypeVar("OptStateType", bound=Any)

LossFn = Callable[[PRNGKey, int, ParamType], Scalar]
LossFnWithData = Callable[[PRNGKey, int, ParamType, DataType], Scalar]
ApplyGradsFn = Callable[
        [PRNGKey, int, ParamType, OptStateType],
        Tuple[ParamType, OptStateType, Scalar]]
ApplyGradsWithDataFn = Callable[
        [PRNGKey, int, ParamType, OptStateType, DataType],
        Tuple[ParamType, OptStateType, Scalar]]
TrainStepFn = Callable[
        [PRNGKey, int, ParamType, OptStateType],
        Tuple[ParamType, OptStateType, Scalar]]
SummarizeFn = Callable[[PRNGKey, ParamType, int], None]


def avg_loss(
        loss_fn: LossFnWithData[ParamType, DataType],
        key: PRNGKey,
        step: int,
        params: ParamType,
        batch: DataType) -> Scalar:
  """Compute the average of a loss function over a batch.

  Meant to be used with functools.partial to supply the loss function. When
  loss_fn is partialed in, this function becomes a LossFnWithData.

  Args:
    loss_fn: A LossFnWithData that accepts a key, parameters, and a single
      datapoint and returns the loss at that datapoint.
    key: A JAX PRNGKey
    params: The parameters to compute the loss at.
    batch: A batch of data.

  Returns:
    The average loss computed over the batch.
  """
  batch_size = jax.tree_util.tree_flatten(batch)[0][0].shape[0]
  losses = jax.vmap(loss_fn, in_axes=(0, None, None, 0))(
          jax.random.split(key, num=batch_size), step, params, batch)
  return jnp.mean(losses)


def apply_grads_with_data(
        loss_fn: LossFnWithData[ParamType, DataType],
        optimizer: optax.GradientTransformation,
        key: PRNGKey,
        step: int,
        params: ParamType,
        opt_state: OptStateType,
        data: DataType
      ) -> Tuple[ParamType, OptStateType, Scalar]:
  """Computes the gradients of a loss function of parameters and data.

   Meant to be used with functools.partial to supply loss_fn and optimizer.
   When loss_fn and optimizer are partialed in, this function becomes an
   ApplyGradsWithDataFn.

   Args:
    loss_fn: A LossFnWithData that accepts a key, parameters, and a single
      datapoint and returns the loss at that datapoint.
    optimizer: An optax optimizer used for applying gradients.
    key: A JAX PRNGKey.
    params: The parameters to compute gradients w.r.t.
    opt_state: The state of the optimizer
    data: Data to supply to the loss function.
   Returns:
     params: The new parameters after applying the gradient update.
     opt_state: The new optimizer state.
     loss_val: The value of the loss
  """
  loss_val, grads = jax.value_and_grad(loss_fn, argnums=2)(key, step, params, data)
  updates, opt_state = optimizer.update(grads, opt_state, params=params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_val


def apply_grads(
        loss_fn: LossFn[ParamType],
        optimizer: optax.GradientTransformation,
        key: PRNGKey,
        step: int,
        params: ParamType,
        opt_state: OptStateType
      ) -> Tuple[ParamType, OptStateType, Scalar]:
  """Computes the gradients of a loss function of parameters and data.

   Meant to be used with functools.partial to supply loss_fn and optimizer.
   When loss_fn and optimizer are partialed in, this function becomes an
   ApplyGradsFn.

   Args:
    loss_fn: A LossFnWithData that accepts a key, parameters, and a single
      datapoint and returns the loss at that datapoint.
    optimizer: An optax optimizer used for applying gradients.
    key: A JAX PRNGKey.
    params: The parameters to compute gradients w.r.t.
    opt_state: The state of the optimizer
   Returns:
     params: The new parameters after applying the gradient update.
     opt_state: The new optimizer state.
     loss_val: The value of the loss
   """
  return apply_grads_with_data(
          lambda k, s, p, _: loss_fn(k, s, p),
          optimizer, key, step, params, opt_state, [])


@dataclass
class TrainStep(Generic[ParamType, OptStateType]):

  train_step_fn: TrainStepFn[ParamType, OptStateType]
  optimizer: optax.GradientTransformation
  num_inner_steps: int = 1
  name: str = "loss"

  def __call__(
          self,
          key: PRNGKey,
          step: int,
          params: ParamType,
          opt_state: OptStateType) -> Tuple[ParamType, OptStateType, Scalar]:
    return self.train_step_fn(key, step, params, opt_state)


def make_train_many_steps(
        loss_fn: LossFn[ParamType],
        optimizer: optax.GradientTransformation,
        num_steps: int,
        name="loss"
      ) -> TrainStep[ParamType, Any]:

  apply_grads_fn = jax.jit(partial(apply_grads, loss_fn, optimizer))

  def train_many_steps(
          key: PRNGKey,
          step: int,
          params: ParamType,
          opt_state: OptStateType) -> Tuple[ParamType, OptStateType, Scalar]:
    loss_val = 0.
    for s in range(num_steps):
      key, subkey = jax.random.split(key)
      params, opt_state, loss_val = apply_grads_fn(subkey, step + s, params, opt_state)
    return params, opt_state, loss_val

  return TrainStep(train_many_steps, optimizer, num_inner_steps=num_steps, name=name)


def make_train_step(
        loss_fn: LossFn[ParamType],
        optimizer: optax.GradientTransformation,
        name: str = "loss"
      ) -> TrainStep[ParamType, Any]:
  """Makes a jitted train step.

  Args:
    loss_fn: A LossFn that will be minimized.
    optimizer: An optax optimizer used to compute gradient updates.
  Returns:
    A jitted train_step that computes the gradients of the loss function
      and applies them to the parameters using optimizer.
  """
  apply_grads_fn = jax.jit(partial(apply_grads, loss_fn, optimizer))
  return TrainStep(apply_grads_fn, optimizer, num_inner_steps=1, name=name)


def make_train_many_steps_with_data(
        loss_fn: LossFnWithData[ParamType, DataType],
        optimizer: optax.GradientTransformation,
        dataset_constructor: Callable[[PRNGKey, ParamType, int], Iterable[DataType]],
        num_steps: int,
        name: str = "loss"
      ) -> TrainStep[ParamType, Any]:

  avg_loss_fn = partial(avg_loss, loss_fn)
  apply_grads_fn = jax.jit(partial(apply_grads_with_data, avg_loss_fn, optimizer))

  def train_many_steps(
          key: PRNGKey,
          step: int,
          params: ParamType,
          opt_state: OptStateType) -> Tuple[ParamType, OptStateType, Scalar]:
    loss_val = 0.
    key, subkey = jax.random.split(key)
    ds = iter(dataset_constructor(key, params, step))
    for s in range(num_steps):
      key, subkey = jax.random.split(key)
      batch = next(ds)
      params, opt_state, loss_val = apply_grads_fn(subkey, step + s, params, opt_state, batch)

    return params, opt_state, loss_val

  return TrainStep(train_many_steps, optimizer, num_inner_steps=num_steps, name=name)


def make_train_step_with_data(
        loss_fn: LossFnWithData[ParamType, DataType],
        optimizer: optax.GradientTransformation,
        dataset: Iterable[DataType],
        name: str = "loss"
      ) -> TrainStep[ParamType, Any]:
  """Makes a jitted train step that accepts data.

  Args:
    loss_fn: A LossFnWithData that will be averaged over each batch.
    optimizer: An optax optimizer used to compute gradient updates.
    dataset: An Iterable that produces batches. next() will be called on iter(dataset)
      at each training step.
  Returns:
    A jitted train_step that requests a batch from iter(dataset), computes the gradients
      of the average loss function, and applies them to the parameters using optimizer.
  """
  avg_loss_fn = partial(avg_loss, loss_fn)
  apply_grads_fn = jax.jit(partial(apply_grads_with_data, avg_loss_fn, optimizer))
  ds = iter(dataset)

  def train_step(
          key: PRNGKey,
          step: int,
          params: ParamType,
          opt_state: OptStateType) -> Tuple[ParamType, OptStateType, Scalar]:
    batch = next(ds)
    return apply_grads_fn(key, step, params, opt_state, batch)

  return TrainStep(train_step, optimizer, num_inner_steps=1, name=name)


def train_alternating(
        key: PRNGKey,
        train_step_fns: List[TrainStep[ParamType, Any]],
        init_params: ParamType,
        num_steps: int = 100,
        summarize_fn: Optional[SummarizeFn[ParamType]] = None,
        summarize_every: int = 100,
        checkpoint_every: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoints_to_keep: int = 3,
        use_wandb: bool = False) -> ParamType:
  """Run training.

  Iteratively runs a set of training steps, logging performance metrics and computing summaries.

  Args:
    key: A JAX PRNGKey.
    train_step_fns: A list of TrainSteps.
    init_params: The initial parameters.
    num_steps: The number of steps to run training for.
    summarize_fn: A function that computes and logs summaries. Must accept a key,
      the current parameters, and the current step.
    summarize_every: The number of steps between calls to summarize_fn.
    checkpoint_every: The number of steps between checkpoints.
    checkpoint_dir: The directory to store checkpoints.
    checkpoints_to_keep: The number of recent checkpoints to keep in the checkpoint directory.
    use_wanbd: Whether to use weights and biases to log performance metrics like steps per
      second and the number of seconds the summary_fn takes.
  Returns:
    The parameters after training for num_steps.
  """
  # Check that summarize_every and checkpoint_every are multiples of the number of inner steps.
  total_inner_steps = sum([ts.num_inner_steps for ts in train_step_fns])
  if summarize_fn is not None:
    assert summarize_every % total_inner_steps == 0, \
            "summarize_every must be a multiple of the total number of inner steps"
  if checkpoint_every is not None:
    assert checkpoint_every % total_inner_steps == 0, \
            "checkpoint_every must be a multiple of the total number of inner steps"

  # Set a dummy summarize_fn if it was not provided
  if summarize_fn is None:
    summarize_fn = lambda *args: None

  # Initialize the parameters and opt_states
  params = init_params
  opt_states = []
  global_start_step = 0
  local_steps = [0] * len(train_step_fns)
  for ts in train_step_fns:
    opt_states.append(ts.optimizer.init(params))

  # Maybe load a checkpoint.
  should_checkpoint = ((checkpoint_dir is not None) and (checkpoint_every is not None))
  if should_checkpoint:
    assert checkpoint_dir is not None
    out = chk.load_latest_checkpoint(checkpoint_dir)
    if out is not None:
      (params, opt_states, local_steps), global_start_step = out
      print(f"Loaded checkpoint at step {global_start_step} from {checkpoint_dir}.")
    else:
      print("Checkpoint not found.")

  # Summarize on the first step.
  print(f"Step {global_start_step}")
  key, subkey = jax.random.split(key)
  summarize_fn(subkey, params, global_start_step)
  # Train.
  step = global_start_step
  while step < num_steps:
    new_opt_states = []
    new_local_steps = []
    loss_vals = []
    steps_per_sec = []
    # Run the train_fns for one step each.
    for opt_state, train_step_fn, local_step in zip(opt_states, train_step_fns, local_steps):
      start_time = timer()
      key, subkey = jax.random.split(key)
      params, new_opt_state, loss_val = train_step_fn(subkey, local_step, params, opt_state)
      new_opt_states.append(new_opt_state)
      loss_vals.append(loss_val)
      new_local_steps.append(local_step + train_step_fn.num_inner_steps)
      step += train_step_fn.num_inner_steps
      sec = timer() - start_time
      steps_per_sec.append(train_step_fn.num_inner_steps / sec)
    opt_states = new_opt_states
    local_steps = new_local_steps
    # Possibly summarize and save a checkpoint.
    if step != global_start_step:
      if step % summarize_every == 0:
        # Print losses.
        print(f"Step {step}")
        for lv, sps, ts in zip(loss_vals, steps_per_sec, train_step_fns):
          print(f"  {ts.name}: {lv:0.3f}, steps/sec: {sps:0.2f}")
        # Compute summaries
        summ_start_time = timer()
        key, subkey = jax.random.split(key)
        summarize_fn(subkey, params, step)
        summ_elapsed_time = timer() - summ_start_time
        print(f"  summary sec: {summ_elapsed_time:0.2f}")
        # Log performance stats.
        if use_wandb:
          wandb.log({
              "train_loss": {ts.name: lv for ts, lv in zip(train_step_fns, loss_vals)},
              "steps_per_sec": {ts.name: sps for ts, sps in zip(train_step_fns, steps_per_sec)},
              "summ_secs": summ_elapsed_time
              }, step=step,
          )
        start_time = timer()
      if should_checkpoint and checkpoint_every is not None and step % checkpoint_every == 0:
        print(f"Saving checkpoint for step {step} at {checkpoint_dir}... ", end="")
        chk.save_checkpoint((params, opt_states, local_steps), step, checkpoint_dir,
                            num_checkpoints_to_keep=checkpoints_to_keep)
        print("Done.")

  return params


def train(key: PRNGKey,
          train_step: TrainStep[ParamType, OptStateType],
          init_params: ParamType,
          num_steps: int = 100,
          summarize_fn: Optional[SummarizeFn[ParamType]] = None,
          summarize_every: int = 100,
          checkpoint_every: Optional[int] = None,
          checkpoint_dir: Optional[str] = None,
          checkpoints_to_keep: int = 3,
          use_wandb: bool = False) -> ParamType:
  """Run training.

  Iteratively runs a training step, logging performance metrics and computing summaries.

  Args:
    key: A JAX PRNGKey.
    train_step: A TrainStep.
    init_params: The initial parameters.
    num_steps: The number of steps to run training for.
    summarize_fn: A function that computes and logs summaries. Must accept a key,
      the current parameters, and the current step.
    summarize_every: The number of steps between calls to summarize_fn.
    checkpoint_every: The number of steps between checkpoints.
    checkpoint_dir: The directory to store checkpoints.
    checkpoints_to_keep: The number of recent checkpoints to keep in the checkpoint directory.
    use_wanbd: Whether to use weights and biases to log performance metrics like steps per
      second and the number of seconds the summary_fn takes.
  Returns:
    The parameters after training for num_steps.
  """
  return train_alternating(key,
                           [train_step],
                           init_params,
                           num_steps=num_steps,
                           summarize_fn=summarize_fn,
                           summarize_every=summarize_every,
                           checkpoint_every=checkpoint_every,
                           checkpoint_dir=checkpoint_dir,
                           checkpoints_to_keep=checkpoints_to_keep,
                           use_wandb=use_wandb)
