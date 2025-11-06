from tensorflow.keras import layers, Model, preprocessing

def load_data(random_seed=42):
    data_dir = './data/raw/all_data'
    img_size = (224, 224)
    batch_size = 32

    total_split = 0.30  # 15% val + 15% test

    train_ds = preprocessing.image_dataset_from_directory(
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

    # Now split temp_ds into 50% val, 50% test
    val_size = 0.5

    val_ds = temp_ds.take(int(len(temp_ds) * val_size))
    test_ds = temp_ds.skip(int(len(temp_ds) * val_size))

    class_names = train_ds.class_names
    print("Classes:", class_names)

    return train_ds, val_ds, test_ds, class_names, img_size