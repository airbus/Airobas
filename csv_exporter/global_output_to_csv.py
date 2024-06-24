import csv

import numpy as np

from verif_pipeline import GlobalVerifOutput, StatusVerif


def dump_csv(global_verif_output: GlobalVerifOutput, file_output: str):
    with open(file_output, "w") as f:
        writer = csv.writer(f)
        # Writing headers
        writer.writerow(
            [
                "Point #",
                "Verification Result",
                "Overall Execution Time",
                "Verification Execution Time",
                "Satisfying assignment (If SAT)",
                "method_index_conclude",
                "tag_index_conclude",
            ]
        )
        # Writing data rows
        for i in range(len(global_verif_output.status)):
            status = ""
            if global_verif_output.status[i] == StatusVerif.VIOLATED:
                status = "sat"
            if global_verif_output.status[i] == StatusVerif.VERIFIED:
                status = "unsat"
            if global_verif_output.status[i] == StatusVerif.TIMEOUT:
                status = "TIMEOUT"
            if global_verif_output.status[i] == StatusVerif.UNKNOWN:
                status = "UNKNOWN"
            writer.writerow(
                [
                    str(i),
                    status,
                    global_verif_output.init_time_per_sample[i]
                    + global_verif_output.verif_time_per_sample[i],
                    global_verif_output.verif_time_per_sample[i],
                    global_verif_output.inputs[i]
                    if global_verif_output.inputs[i] is not None
                    else [],
                    global_verif_output.index_block_that_concluded[i]
                    if status not in {"TIMEOUT", "UNKNOWN"}
                    else None,
                    global_verif_output.methods[
                        global_verif_output.index_block_that_concluded[i]
                    ]
                    if status not in {"TIMEOUT", "UNKNOWN"}
                    else None,
                ]
            )
