from tensorflow.keras import layers, Model, preprocessing
import tensorflow as tf

def get_class_counts(ds, num_classes):
    counts = [0] * num_classes
    for _, y in ds:
        labels = tf.argmax(y, axis=1).numpy()
        for label in labels:
            counts[label] += 1
    return counts

def load_data(random_seed=42, balanced_augmentation=True, minority_threshold=0.75):
    data_dir = './data/raw/all_data'
    img_size = (224, 224)
    batch_size = 32
    total_split = 0.30   # 15% val + 15% test

    # ---------------------------------------------------
    # 1. Load training + temporary validation dataset
    # ---------------------------------------------------
    train_ds_raw = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=total_split,
        subset='training',
        seed=random_seed,
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        crop_to_aspect_ratio=True
    )

    temp_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=total_split,
        subset='validation',
        seed=random_seed,
        label_mode='categorical',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        crop_to_aspect_ratio=True
    )

    # Split validation set into val + test
    val_size = 0.5
    val_ds = temp_ds.take(int(len(temp_ds) * val_size))
    test_ds = temp_ds.skip(int(len(temp_ds) * val_size))

    # Class names
    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    print("\nClasses:", class_names)

    # ---------------------------------------------------
    # If augmentation is disabled → return normal dataset
    # ---------------------------------------------------
    if not balanced_augmentation:
        print("\n⚠ Balanced minority-only augmentation is OFF")
        print("Returning raw (unbalanced) dataset.\n")

        # Prefetch for performance
        return (
            train_ds_raw.prefetch(tf.data.AUTOTUNE),
            val_ds.prefetch(tf.data.AUTOTUNE),
            test_ds.prefetch(tf.data.AUTOTUNE),
            class_names,
            img_size
        )

    # ---------------------------------------------------
    # 2. Balanced augmentation mode ON
    # ---------------------------------------------------
    print("\n✅ Balanced minority-only augmentation is ON")

    # Count images per class
    print("\nCounting class frequencies...")
    class_counts = get_class_counts(train_ds_raw, num_classes)
    max_count = max(class_counts)

    for cname, count in zip(class_names, class_counts):
        print(f"{cname}: {count}")

    # ---------------------------------------------------
    # 3. Define augmentation for MINORITY classes ONLY
    # ---------------------------------------------------
    minority_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])

    # ---------------------------------------------------
    # 4. Create minority-augmented and majority datasets
    # ---------------------------------------------------
    train_ds_unbatched = train_ds_raw.unbatch()

    minority_datasets = []
    majority_datasets = []

    for class_idx, count in enumerate(class_counts):
        class_subset = train_ds_unbatched.filter(
            lambda x, y, idx=class_idx: tf.argmax(y) == idx
        )

        # Identify minority classes
        if count < minority_threshold * max_count:
            repeat_factor = int(max_count / count)

            print(f"\nMinority class: {class_names[class_idx]} "
                  f"(count={count}) → repeating {repeat_factor}× with augmentation")

            augmented = class_subset.map(
                lambda x, y: (minority_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            ).repeat(repeat_factor)

            minority_datasets.append(augmented)

        else:
            print(f"\nMajority class: {class_names[class_idx]} (count={count})")
            majority_datasets.append(class_subset)

    # ---------------------------------------------------
    # 5. Combine into final balanced dataset
    # ---------------------------------------------------
    final_train_ds = tf.data.Dataset.sample_from_datasets(
        minority_datasets + majority_datasets
    )

    final_train_ds = final_train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    print("\n✔ Balanced + minority-only augmented training dataset ready.")

    return final_train_ds, val_ds, test_ds, class_names, img_size