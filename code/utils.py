def extract_val_pearson(file_name):
    # Extract the val_pearson value from the file name
    val_pearson = float(file_name.split("val_pearson=")[1].split(".ckpt")[0])
    return val_pearson