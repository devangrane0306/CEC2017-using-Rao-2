import os
import csv


def save_to_csv(func_id, dimension, best, worst, mean, std_dev):

    folder = "results/logs"
    file_path = os.path.join(folder, "results.csv")

    # create folder if not exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # write header only once
        if not file_exists:
            writer.writerow([
                "Function",
                "Dimension",
                "Best",
                "Worst",
                "Mean",
                "StdDev"
            ])

        writer.writerow([
            "F" + str(func_id),
            dimension,
            best,
            worst,
            mean,
            std_dev
        ])