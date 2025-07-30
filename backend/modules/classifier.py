import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    roc_curve,
    auc,
)

def average_precision(labels, scores):
    #labels = np.array(labels)
    #scores = np.array(scores)
    idx = np.argsort(scores) #[:,0]
    idx = np.flip(idx)
    scores = np.take_along_axis(scores, idx, axis=-1) #[:,0]
    labels = np.take_along_axis(labels, idx, axis=-1) #[:,0]
    pr_curve = np.cumsum(labels, axis=-1) / (np.arange(labels.shape[-1]) + 1)
    ap = np.sum(pr_curve * labels, axis=-1) / np.maximum(np.sum(labels, axis=-1), 1.0)
    return ap

def get_AUC(labels, scores):
    AUCs = []
    for i in range(labels.shape[1]):
        fpr, tpr, thresholds = roc_curve(labels[:, i], scores[:, i])
        AUCs.append(auc(fpr, tpr))
    return {
        "macro": np.mean(AUCs),
        "individual": AUCs,
    }

def cmap(labels, scores, threshold):  #
    """Class mean average precision."""
    labels = np.array(labels)
    scores = np.array(scores)
    class_aps = []
    for i in range(labels.shape[1]):
        class_aps.append(average_precision(labels[:, i], scores[:, i]))

    col_sums = labels.sum(axis=0)
    # vali_aps = [
    #    class_aps[i] if sum > 0 else "NA" for i, sum in enumerate(list(col_sums))
    # ]

    # Filter out the "NA" values and calculate the mean of the remaining values
    valid_aps = [ap for i, ap in enumerate(class_aps) if col_sums[i] > threshold]

    mask = np.sum(labels, axis=0) > threshold
    macro_cmap = np.mean(class_aps, where=mask)
    # macro_cmap = np.mean(valid_aps, where=mask)
    return {
        "macro": macro_cmap,
        "mask": mask,
        "individual": valid_aps,
    }


def evaluate_model(CLASSIFIER, DATA, LABELS, PREDS, CASE_STUDY, K, REP):
    eval_dataset = tf.data.experimental.from_list(list(PREDS))
    max_preds = eval_dataset.map(
        CLASSIFIER,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    max_preds = [np.amax(max_pred.numpy(), axis=0) for max_pred in max_preds]
    np_max_pred = np.array(tf.sigmoid(max_preds))
    cmaps = cmap(LABELS, np_max_pred)
    AUCs = get_AUC(LABELS, np_max_pred)
    results = {}
    results["case_study"] = CASE_STUDY
    results["k"] = K
    results["rep"] = REP
    results["dataset"] = DATA
    results["AUC_macro"] = AUCs["macro"]
    results["AUCs"] = AUCs["individual"]
    results["mAP"] = cmaps["macro"]
    results["APs"] = cmaps["individual"]
    return results



def build_model(input_dim, output_dim, type):
    model_template = tf.keras.Sequential()
    model_template.add(tf.keras.Input(shape=([input_dim]), dtype=tf.float32))
    activation = "relu"
    # activation = "leaky_relu"
    # activation = "silu"
    dropout = 0.3
    # model_template.add(tf.keras.layers.BatchNormalization())
    if type == 2:
        model_template.add(tf.keras.layers.Dropout(dropout))
    if type == 3:
        model_template.add(tf.keras.layers.Dense(1280, activation=activation))
    if type == 4:
        model_template.add(tf.keras.layers.Dense(1280, activation=activation))
        model_template.add(tf.keras.layers.Dropout(dropout))
    if type == 5:
        model_template.add(tf.keras.layers.Dense(2 * 1024, activation=activation))
    if type == 6:
        # model_template.add(tf.keras.layers.Dropout(dropout))
        model_template.add(tf.keras.layers.Dense(2 * 1024, activation=activation))
        model_template.add(tf.keras.layers.Dropout(dropout))
    if type == 7:
        model_template.add(tf.keras.layers.Dense(512, activation=activation))
    if type == 8:
        # model_template.add(tf.keras.layers.Dropout(dropout))
        model_template.add(tf.keras.layers.Dense(512, activation=activation))
        model_template.add(tf.keras.layers.Dropout(dropout))
    model_template.add(
        tf.keras.layers.Dense(
            output_dim
        )  # , kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)
    )
    return model_template

def bce_loss(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    is_labeled_mask: tf.Tensor,
    weak_neg_weight: float,
) -> tf.Tensor:
  """Binary cross entropy loss from logits with weak negative weights."""
  y_true = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
  log_p = tf.math.log_sigmoid(logits)
  log_not_p = tf.math.log_sigmoid(-logits)
  # optax sigmoid_binary_cross_entropy:
  # -labels * log_p - (1.0 - labels) * log_not_p
  raw_bce = -y_true * log_p - (1.0 - y_true) * log_not_p
  is_labeled_mask = tf.cast(is_labeled_mask, dtype=logits.dtype)
  weights = (1.0 - is_labeled_mask) * weak_neg_weight + is_labeled_mask
  return tf.reduce_mean(raw_bce * weights)
  
def fit_w_tape(
    embeddings,
    labels,
    eval_embeddings,
    eval_labels,
    N_STEPS,
    BATCH_SIZE,
    LEARNING_RATE,
    TYPE,
    save_path,
    verbose,
    label_strength=None,
    eval_label_strength=None,
    weak_neg_weight=0.05,
):
    # embeddings = np.array(X_train).squeeze()
    # labels = np.array(y_train)
    # eval_embeddings =  np.array(X_test).squeeze()
    # eval_labels = np.array(X_train).squeeze()
    # N_STEPS = 1000
    # BATCH_SIZE = 128
    # LEARNING_RATE = 0.1
    # TYPE = 2

    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    if np.array(labels).shape[0] < batch_size:
        batch_size = np.array(labels).shape[0]
    
    # Create datasets for embeddings and labels
    file_path_ds = tf.data.Dataset.from_tensor_slices(embeddings)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    # Handle label strength (weak/strong labeling)
    if label_strength is not None:
        # Convert label strength to is_labeled_mask (1 for strong labels, 0 for weak labels)
        is_labeled_mask = tf.cast(tf.convert_to_tensor(label_strength), tf.float32)
        mask_ds = tf.data.Dataset.from_tensor_slices(is_labeled_mask)
        ds = tf.data.Dataset.zip((file_path_ds, label_ds, mask_ds))
    else:
        # If no label strength provided, assume all labels are strong (is_labeled_mask = 1)
        default_mask = tf.ones(tf.shape(labels)[0], dtype=tf.float32)
        mask_ds = tf.data.Dataset.from_tensor_slices(default_mask)
        ds = tf.data.Dataset.zip((file_path_ds, label_ds, mask_ds))
    
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    classifier_model = build_model(embeddings.shape[1], labels.shape[1], TYPE)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
    #    apply_class_balancing=True, gamma=3, from_logits=True
    # )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(x, y, mask):
        with tf.GradientTape() as tape:
            logits = classifier_model(x, training=True)
            # Use the custom bce_loss function with weak/strong labeling
            loss_value = bce_loss(y, logits, mask, weak_neg_weight)
            grads = tape.gradient(loss_value, classifier_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, classifier_model.trainable_weights))
        return loss_value

    def test_step(x, y, eval_mask):
        # Using mode.predict() results in a memory leak if it is iteratively used.
        logits = classifier_model(x, training=False)
        # Use the custom bce_loss function for validation as well
        val_loss = bce_loss(y, logits, eval_mask, weak_neg_weight)
        preds = tf.sigmoid(np.array(logits))
        cmaps = cmap(y, np.array(preds), 0)
        return val_loss, cmaps

    # Prepare evaluation mask
    if eval_label_strength is not None:
        eval_mask = tf.cast(tf.convert_to_tensor(eval_label_strength), tf.float32)
    else:
        # If no evaluation label strength provided, assume all labels are strong
        eval_mask = tf.ones(tf.shape(eval_labels)[0], dtype=tf.float32)
    
    train_losses = []
    val_losses = []
    cmaps = []
    best_val_loss = 1000.0
    best = 0.0
    patience = 5000
    lr_wait = 0
    wait = 0
    lr_reduce_patience = 1000
    lr_redux = 0.5  # 0.9
    for step, (x_batch_train, y_batch_train, mask_batch_train) in enumerate(ds):
        if step == N_STEPS:
            break
        loss_value = train_step(tf.squeeze(x_batch_train), y_batch_train, mask_batch_train)
        train_losses.append(loss_value)
        val_loss, score = test_step(eval_embeddings, eval_labels, eval_mask)
        val_losses.append(val_loss)
        cmaps.append(score["macro"])
        # print(score["macro"])

        # implementing early stopping
        # Saving checkpoints
        if step > 0 and score["macro"] > best:
            if verbose:
                print(
                    f'Macro cMAP of best fit {score["macro"]} obtained on step: {step} val loss: {val_loss}'
                )
            best = score["macro"]
            score_indiv = score["individual"]
            # Ensure the directory exists before saving
            import os
            from pathlib import Path
            save_dir = Path(save_path).parent
            os.makedirs(save_dir, exist_ok=True)
            classifier_model.save(save_path)
            #classifier_model.save(f"./checkpoints/{save_name}.h5py")
        # Early stopping
        wait += 1
        # Reduce learning rate
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            lr_wait = 0
        else:
            lr_wait += 1
            if lr_wait >= lr_reduce_patience:
                if verbose:
                    print(
                        f"Reducing learning rate: from {learning_rate} to {learning_rate*lr_redux} on step {step}"  # learning_rate*0.9
                    )
                learning_rate = learning_rate * lr_redux
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                lr_wait = 0
        if wait >= patience or learning_rate < 0.00001:
            break
    print(f"Final loss: {loss_value}")
    print(
        f"Macro cMAP of best fit {best}"  # obtained on step: {step} val loss: {val_loss}'
    )
    return classifier_model, train_losses, val_losses, cmaps
