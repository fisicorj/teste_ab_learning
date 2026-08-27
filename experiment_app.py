"""
Servidor de Teste A/B "ao vivo" para uso em sala de aula
==========================================================

Cada aluno abre uma URL no navegador dele. O servidor:

1. Identifica o aluno (pelo nome/matrícula que ele digita, ou por um cookie
   anônimo, se preferir).
2. Usa o MESMO hashing consistente do script anterior (ab_testing_demo.py)
   para decidir se ele cai no Grupo A (controle) ou Grupo B (tratamento).
   -> Cada aluno SEMPRE cai no mesmo grupo, mesmo se recarregar a página.
3. Mostra uma página diferente para cada grupo (aqui: cor/texto do botão
   "Comprar").
4. Quando o aluno clica no botão, registra um evento de "conversão" em um
   banco SQLite, junto com o tempo que ele levou para clicar.

Como rodar:
    pip install flask
    python experiment_app.py

Isso sobe um servidor em http://0.0.0.0:5000

- Se os alunos estiverem na MESMA rede Wi-Fi que você: descubra seu IP local
  (`ipconfig` no Windows / `ifconfig` ou `ip a` no Mac/Linux, procure por algo
  tipo 192.168.x.x) e peça para eles acessarem http://SEU_IP:5000
- Se os alunos estiverem em redes diferentes / remotos: rode `ngrok http 5000`
  (https://ngrok.com) e mande o link público que ele gera, ou implante em um
  serviço como Render, Railway ou PythonAnywhere.

Os dados de cada aluno ficam salvos em `experiment.db` (SQLite), no mesmo
diretório. Rode `analyze_real_data.py` depois para ver o resultado real.
"""

import hashlib
import sqlite3
import time
from datetime import datetime, timezone

from flask import Flask, g, redirect, render_template_string, request, url_for

# Reaproveita as funções estatísticas já escritas e testadas em ab_testing_demo.py
from ab_testing_demo import two_proportion_z_test, welch_t_test

app = Flask(__name__)

DB_PATH = "experiment.db"
EXPERIMENT_NAME = "cor_do_botao_comprar"  # troque para identificar seu experimento
SPLIT = 0.5  # % do tráfego que vai para o grupo B


# ---------------------------------------------------------------------------
# Banco de dados
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS assignments (
            student_id TEXT PRIMARY KEY,
            experiment TEXT NOT NULL,
            group_name TEXT NOT NULL,
            assigned_at TEXT NOT NULL,
            page_loaded_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            experiment TEXT NOT NULL,
            event_type TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            seconds_since_load REAL
        )
        """
    )
    conn.commit()
    conn.close()


# Garante que as tabelas existam sempre que o módulo é carregado, não só
# quando rodado com `python experiment_app.py` (cobre `flask run`, gunicorn, etc.)
init_db()


# ---------------------------------------------------------------------------
# Atribuição de grupo (mesma lógica do ab_testing_demo.py)
# ---------------------------------------------------------------------------

def assign_group(student_id: str) -> str:
    key = f"{EXPERIMENT_NAME}:{student_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "B" if bucket < SPLIT else "A"


def get_or_create_assignment(student_id: str) -> tuple[str, float]:
    """Retorna (grupo, page_loaded_at). Garante que o aluno sempre veja o
    mesmo grupo, mesmo recarregando a página."""
    db = get_db()
    row = db.execute(
        "SELECT group_name, page_loaded_at FROM assignments WHERE student_id = ? AND experiment = ?",
        (student_id, EXPERIMENT_NAME),
    ).fetchone()
    if row is not None:
        return row["group_name"], row["page_loaded_at"]

    group = assign_group(student_id)
    now = time.time()
    db.execute(
        "INSERT INTO assignments (student_id, experiment, group_name, assigned_at, page_loaded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (student_id, EXPERIMENT_NAME, group, datetime.now(timezone.utc).isoformat(), now),
    )
    db.commit()
    return group, now


# ---------------------------------------------------------------------------
# Páginas (as duas variantes do experimento)
# ---------------------------------------------------------------------------

LOGIN_PAGE = """
<!doctype html>
<title>Entrar no experimento</title>
<h2>Teste A/B - identifique-se</h2>
<form action="{{ url_for('experiment') }}" method="get">
  <label>Seu nome ou matrícula: <input name="student_id" required></label>
  <button type="submit">Entrar</button>
</form>
"""

VARIANT_PAGE = """
<!doctype html>
<title>Loja Exemplo</title>
<style>
  body { font-family: sans-serif; max-width: 480px; margin: 80px auto; text-align: center; }
  .btn {
    padding: 16px 32px; font-size: 18px; border: none; border-radius: 8px;
    cursor: pointer; color: white;
  }
  .btn-a { background: #6b7280; }   /* controle: cinza */
  .btn-b { background: #16a34a; }   /* tratamento: verde */
</style>
<h1>Produto Incrível</h1>
<p>Você foi direcionado para o Grupo {{ group }}.</p>
<button class="btn {{ 'btn-b' if group == 'B' else 'btn-a' }}" onclick="converter()">
  {{ 'Aproveitar agora!' if group == 'B' else 'Comprar' }}
</button>
<p id="msg"></p>
<script>
  async function converter() {
    const resp = await fetch("{{ url_for('convert') }}", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({student_id: "{{ student_id }}"})
    });
    document.getElementById("msg").innerText = resp.ok ? "Obrigado! Conversão registrada." : "Erro ao registrar.";
  }
</script>
"""


@app.route("/")
def index():
    return render_template_string(LOGIN_PAGE)


@app.route("/experiment")
def experiment():
    student_id = request.args.get("student_id", "").strip()
    if not student_id:
        return redirect(url_for("index"))

    group, _ = get_or_create_assignment(student_id)
    return render_template_string(VARIANT_PAGE, group=group, student_id=student_id)


@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json(force=True)
    student_id = data["student_id"]

    db = get_db()
    row = db.execute(
        "SELECT page_loaded_at FROM assignments WHERE student_id = ? AND experiment = ?",
        (student_id, EXPERIMENT_NAME),
    ).fetchone()
    seconds_since_load = (time.time() - row["page_loaded_at"]) if row else None

    db.execute(
        "INSERT INTO events (student_id, experiment, event_type, occurred_at, seconds_since_load) "
        "VALUES (?, ?, 'conversion', ?, ?)",
        (student_id, EXPERIMENT_NAME, datetime.now(timezone.utc).isoformat(), seconds_since_load),
    )
    db.commit()
    return {"status": "ok"}


RESULTS_PAGE = """
<!doctype html>
<title>Resultados do experimento</title>
<meta http-equiv="refresh" content="15">
<style>
  body { font-family: sans-serif; max-width: 640px; margin: 40px auto; color: #111; }
  h1 { font-size: 22px; }
  h2 { font-size: 17px; margin-top: 32px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  td, th { text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; }
  .ok { color: #16a34a; font-weight: bold; }
  .no { color: #6b7280; }
  .warn { background: #fef3c7; padding: 12px; border-radius: 6px; }
</style>
<h1>Resultados: {{ experiment }}</h1>
<p>Amostra atual: {{ control_total }} no grupo A, {{ treatment_total }} no grupo B.
(a página se atualiza sozinha a cada 15s)</p>

{% if warning %}
  <p class="warn">{{ warning }}</p>
{% else %}
  <h2>Taxa de conversão</h2>
  <table>
    <tr><th></th><th>Grupo A (controle)</th><th>Grupo B (tratamento)</th></tr>
    <tr><td>Taxa</td><td>{{ '%.2f'|format(prop.rate_control*100) }}%</td><td>{{ '%.2f'|format(prop.rate_treatment*100) }}%</td></tr>
    <tr><td>Lift absoluto</td><td colspan="2">{{ '%+.2f'|format(prop.absolute_lift*100) }} p.p.</td></tr>
    <tr><td>Lift relativo</td><td colspan="2">{{ '%+.1f'|format(prop.relative_lift*100) }}%</td></tr>
    <tr><td>IC 95%</td><td colspan="2">[{{ '%+.2f'|format(prop.ci_95[0]*100) }}%, {{ '%+.2f'|format(prop.ci_95[1]*100) }}%]</td></tr>
    <tr><td>p-valor</td><td colspan="2">{{ '%.4f'|format(prop.p_value) }}</td></tr>
    <tr><td>Resultado</td><td colspan="2" class="{{ 'ok' if prop.significant else 'no' }}">
        {{ 'SIGNIFICATIVO ✅' if prop.significant else 'não significativo' }}</td></tr>
  </table>

  <h2>Tempo até a conversão (segundos)</h2>
  {% if mean_available %}
  <table>
    <tr><th></th><th>Grupo A (controle)</th><th>Grupo B (tratamento)</th></tr>
    <tr><td>Média</td><td>{{ '%.1f'|format(mean.mean_control) }}s</td><td>{{ '%.1f'|format(mean.mean_treatment) }}s</td></tr>
    <tr><td>Lift absoluto</td><td colspan="2">{{ '%+.1f'|format(mean.absolute_lift) }}s</td></tr>
    <tr><td>IC 95%</td><td colspan="2">[{{ '%+.1f'|format(mean.ci_95[0]) }}s, {{ '%+.1f'|format(mean.ci_95[1]) }}s]</td></tr>
    <tr><td>p-valor</td><td colspan="2">{{ '%.4f'|format(mean.p_value) }}</td></tr>
    <tr><td>Resultado</td><td colspan="2" class="{{ 'ok' if mean.significant else 'no' }}">
        {{ 'SIGNIFICATIVO ✅' if mean.significant else 'não significativo' }}</td></tr>
  </table>
  {% else %}
    <p class="warn">Poucos cliques registrados ainda para calcular essa métrica.</p>
  {% endif %}
{% endif %}
"""


def compute_live_results(experiment_name: str):
    """Lê assignments/events do banco (via get_db, dentro do request atual) e
    calcula os mesmos testes estatísticos do ab_testing_demo.py."""
    import numpy as np

    db = get_db()
    assignments = db.execute(
        "SELECT student_id, group_name FROM assignments WHERE experiment = ?",
        (experiment_name,),
    ).fetchall()
    conversions = db.execute(
        "SELECT student_id, seconds_since_load FROM events "
        "WHERE experiment = ? AND event_type = 'conversion'",
        (experiment_name,),
    ).fetchall()
    converted = {row["student_id"]: row["seconds_since_load"] for row in conversions}

    control_total = treatment_total = 0
    control_conversions = treatment_conversions = 0
    control_times, treatment_times = [], []

    for row in assignments:
        sid, group = row["student_id"], row["group_name"]
        is_converted = sid in converted
        if group == "A":
            control_total += 1
            if is_converted:
                control_conversions += 1
                if converted[sid] is not None:
                    control_times.append(converted[sid])
        else:
            treatment_total += 1
            if is_converted:
                treatment_conversions += 1
                if converted[sid] is not None:
                    treatment_times.append(converted[sid])

    return {
        "control_total": control_total,
        "treatment_total": treatment_total,
        "control_conversions": control_conversions,
        "treatment_conversions": treatment_conversions,
        "control_times": np.array(control_times, dtype=float),
        "treatment_times": np.array(treatment_times, dtype=float),
    }


@app.route("/resultados")
def resultados():
    r = compute_live_results(EXPERIMENT_NAME)

    if r["control_total"] == 0 or r["treatment_total"] == 0:
        return render_template_string(
            RESULTS_PAGE,
            experiment=EXPERIMENT_NAME,
            control_total=r["control_total"],
            treatment_total=r["treatment_total"],
            warning="Ainda não há alunos registrados nos dois grupos. Peça para mais alunos acessarem /experiment.",
        )

    prop = two_proportion_z_test(
        r["control_conversions"], r["control_total"],
        r["treatment_conversions"], r["treatment_total"],
    )

    mean_available = len(r["control_times"]) >= 2 and len(r["treatment_times"]) >= 2
    mean = welch_t_test(r["control_times"], r["treatment_times"]) if mean_available else None

    return render_template_string(
        RESULTS_PAGE,
        experiment=EXPERIMENT_NAME,
        control_total=r["control_total"],
        treatment_total=r["treatment_total"],
        warning=None,
        prop=prop,
        mean=mean,
        mean_available=mean_available,
    )


if __name__ == "__main__":
    print(f"Experimento '{EXPERIMENT_NAME}' rodando. Peça para os alunos acessarem:")
    print("  http://SEU_IP_LOCAL:5000  (mesma rede Wi-Fi)")
    print("Acompanhe os resultados ao vivo em:")
    print("  http://SEU_IP_LOCAL:5000/resultados")
    app.run(host="0.0.0.0", port=5000, debug=True)
