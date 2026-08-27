"""
Analisa os dados REAIS coletados pelo experiment_app.py (banco experiment.db)
usando as mesmas funções estatísticas de ab_testing_demo.py.

Rode depois que os alunos tiverem acessado o sistema e clicado (ou não) no
botão:
    python analyze_real_data.py
"""

import sqlite3

import numpy as np

# Reaproveita as funções estatísticas já escritas e testadas.
from ab_testing_demo import ExperimentData, print_report

DB_PATH = "experiment.db"
EXPERIMENT_NAME = "cor_do_botao_comprar"


def load_experiment_data(db_path: str, experiment_name: str) -> ExperimentData:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    assignments = conn.execute(
        "SELECT student_id, group_name FROM assignments WHERE experiment = ?",
        (experiment_name,),
    ).fetchall()

    conversions = conn.execute(
        "SELECT student_id, seconds_since_load FROM events "
        "WHERE experiment = ? AND event_type = 'conversion'",
        (experiment_name,),
    ).fetchall()
    conn.close()

    converted_ids = {row["student_id"]: row["seconds_since_load"] for row in conversions}

    control_total = treatment_total = 0
    control_conversions = treatment_conversions = 0
    control_times, treatment_times = [], []

    for row in assignments:
        sid, group = row["student_id"], row["group_name"]
        converted = sid in converted_ids

        if group == "A":
            control_total += 1
            if converted:
                control_conversions += 1
                if converted_ids[sid] is not None:
                    control_times.append(converted_ids[sid])
        else:
            treatment_total += 1
            if converted:
                treatment_conversions += 1
                if converted_ids[sid] is not None:
                    treatment_times.append(converted_ids[sid])

    if control_total == 0 or treatment_total == 0:
        raise RuntimeError(
            "Ainda não há dados suficientes nos dois grupos. "
            "Peça para mais alunos acessarem o sistema."
        )
    if len(control_times) < 2 or len(treatment_times) < 2:
        # Métrica de tempo até conversão exige pelo menos alguns cliques por grupo;
        # se faltar dado, preenche com um placeholder para não quebrar o teste t.
        print("[aviso] poucos cliques registrados para a métrica de tempo até conversão.")
        control_times = control_times or [0.0, 0.0]
        treatment_times = treatment_times or [0.0, 0.0]

    return ExperimentData(
        control_conversions=control_conversions,
        control_total=control_total,
        treatment_conversions=treatment_conversions,
        treatment_total=treatment_total,
        control_session_time=np.array(control_times, dtype=float),
        treatment_session_time=np.array(treatment_times, dtype=float),
    )


if __name__ == "__main__":
    dados = load_experiment_data(DB_PATH, EXPERIMENT_NAME)
    print_report(dados, alpha=0.05)
