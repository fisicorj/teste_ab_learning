"""
Demonstração de Teste A/B em Engenharia de Software
=====================================================

Este script mostra, de ponta a ponta, como um teste A/B costuma ser
implementado em um sistema real:

1. Atribuição de usuários aos grupos (A = controle, B = tratamento)
   usando hashing consistente (o mesmo usuário sempre cai no mesmo grupo,
   mesmo em requisições diferentes — é assim que feature flags funcionam
   na prática, ex: LaunchDarkly, Optimizely, GrowthBook).
2. Simulação de dados de um experimento (ex: taxa de conversão e tempo
   gasto na página).
3. Cálculo do tamanho de amostra necessário antes de rodar o teste.
4. Análise estatística dos resultados:
   - Teste Z para duas proporções (ex: taxa de clique/conversão)
   - Teste t para métricas contínuas (ex: tempo de sessão, receita)
   - Intervalo de confiança e p-valor
5. Um relatório final legível, como o que você mandaria para um PM.

Dependências: numpy, scipy
    pip install numpy scipy
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Atribuição de grupos (feature flag / bucketing)
# ---------------------------------------------------------------------------

class Group(str, Enum):
    CONTROL = "A"
    TREATMENT = "B"


class ExperimentAssigner:
    """
    Atribui usuários a grupos A/B de forma determinística e estável.

    Em produção você NÃO quer sortear o grupo a cada requisição — o mesmo
    usuário precisa sempre cair no mesmo grupo. A técnica padrão é fazer
    hash(experiment_id + user_id) e usar o resultado para decidir o bucket.
    """

    def __init__(self, experiment_name: str, split: float = 0.5, salt: str = ""):
        """
        experiment_name: identifica o experimento (evita colisão entre testes).
        split: fração do tráfego que vai para o grupo B (tratamento).
        salt: permite "re-randomizar" sem mudar o nome do experimento.
        """
        if not 0 < split < 1:
            raise ValueError("split deve estar entre 0 e 1")
        self.experiment_name = experiment_name
        self.split = split
        self.salt = salt

    def assign(self, user_id: str) -> Group:
        key = f"{self.experiment_name}:{self.salt}:{user_id}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        # Usa os primeiros 8 hex chars como inteiro e normaliza para [0, 1)
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return Group.TREATMENT if bucket < self.split else Group.CONTROL


# ---------------------------------------------------------------------------
# 2. Cálculo de tamanho de amostra (antes de rodar o teste!)
# ---------------------------------------------------------------------------

def sample_size_for_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Calcula o tamanho de amostra por grupo necessário para detectar uma
    diferença mínima (MDE) entre duas proporções, com significância `alpha`
    e poder estatístico `power`.

    Exemplo: se a taxa de conversão atual é 10% e você quer detectar um
    aumento de 2 pontos percentuais (para 12%), com alpha=0.05 e power=0.8.
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_avg = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (
        z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2

    return math.ceil(numerator / denominator)


# ---------------------------------------------------------------------------
# 3. Simulação de dados do experimento
# ---------------------------------------------------------------------------

@dataclass
class ExperimentData:
    control_conversions: int
    control_total: int
    treatment_conversions: int
    treatment_total: int
    control_session_time: np.ndarray = field(repr=False)
    treatment_session_time: np.ndarray = field(repr=False)


def simulate_experiment(
    n_per_group: int,
    control_conversion_rate: float = 0.10,
    treatment_conversion_rate: float = 0.12,
    control_session_mean: float = 120.0,
    treatment_session_mean: float = 128.0,
    session_std: float = 40.0,
    seed: int = 42,
) -> ExperimentData:
    """
    Gera dados sintéticos de um experimento: conversão (binária) e
    tempo de sessão em segundos (contínua), para os grupos A e B.
    """
    rng = np.random.default_rng(seed)

    control_conversions = rng.binomial(1, control_conversion_rate, n_per_group)
    treatment_conversions = rng.binomial(1, treatment_conversion_rate, n_per_group)

    control_session_time = rng.normal(control_session_mean, session_std, n_per_group)
    treatment_session_time = rng.normal(treatment_session_mean, session_std, n_per_group)

    return ExperimentData(
        control_conversions=int(control_conversions.sum()),
        control_total=n_per_group,
        treatment_conversions=int(treatment_conversions.sum()),
        treatment_total=n_per_group,
        control_session_time=control_session_time,
        treatment_session_time=treatment_session_time,
    )


# ---------------------------------------------------------------------------
# 4. Análise estatística
# ---------------------------------------------------------------------------

@dataclass
class ProportionTestResult:
    rate_control: float
    rate_treatment: float
    absolute_lift: float
    relative_lift: float
    z_stat: float
    p_value: float
    ci_95: tuple[float, float]
    significant: bool


def two_proportion_z_test(
    conversions_a: int, total_a: int, conversions_b: int, total_b: int, alpha: float = 0.05
) -> ProportionTestResult:
    """Teste Z para diferença entre duas proporções (ex: taxa de conversão)."""
    p_a = conversions_a / total_a
    p_b = conversions_b / total_b

    p_pool = (conversions_a + conversions_b) / (total_a + total_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))

    z_stat = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # bicaudal

    # IC 95% para a diferença (p_b - p_a), usando erro padrão não-pooled
    se_diff = math.sqrt(p_a * (1 - p_a) / total_a + p_b * (1 - p_b) / total_b)
    diff = p_b - p_a
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    return ProportionTestResult(
        rate_control=p_a,
        rate_treatment=p_b,
        absolute_lift=diff,
        relative_lift=(diff / p_a) if p_a > 0 else float("nan"),
        z_stat=z_stat,
        p_value=p_value,
        ci_95=ci,
        significant=p_value < alpha,
    )


@dataclass
class MeanTestResult:
    mean_control: float
    mean_treatment: float
    absolute_lift: float
    relative_lift: float
    t_stat: float
    p_value: float
    ci_95: tuple[float, float]
    significant: bool


def welch_t_test(
    sample_a: np.ndarray, sample_b: np.ndarray, alpha: float = 0.05
) -> MeanTestResult:
    """
    Teste t de Welch para diferença de médias (não assume variâncias iguais
    — é o padrão recomendado para testes A/B, já que os grupos raramente
    têm variância idêntica).
    """
    mean_a, mean_b = sample_a.mean(), sample_b.mean()
    t_stat, p_value = stats.ttest_ind(sample_b, sample_a, equal_var=False)

    se_diff = math.sqrt(sample_a.var(ddof=1) / len(sample_a) + sample_b.var(ddof=1) / len(sample_b))
    dof = len(sample_a) + len(sample_b) - 2  # aproximação simples para o IC
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    diff = mean_b - mean_a
    ci = (diff - t_crit * se_diff, diff + t_crit * se_diff)

    return MeanTestResult(
        mean_control=mean_a,
        mean_treatment=mean_b,
        absolute_lift=diff,
        relative_lift=(diff / mean_a) if mean_a != 0 else float("nan"),
        t_stat=t_stat,
        p_value=p_value,
        ci_95=ci,
        significant=p_value < alpha,
    )


# ---------------------------------------------------------------------------
# 5. Relatório final
# ---------------------------------------------------------------------------

def print_report(data: ExperimentData, alpha: float = 0.05) -> None:
    print("=" * 70)
    print("RELATÓRIO DO TESTE A/B")
    print("=" * 70)

    print(f"\nAmostra: {data.control_total} usuários no grupo A (controle), "
          f"{data.treatment_total} no grupo B (tratamento)\n")

    # --- Métrica 1: taxa de conversão (binária) ---
    prop_result = two_proportion_z_test(
        data.control_conversions, data.control_total,
        data.treatment_conversions, data.treatment_total,
        alpha=alpha,
    )
    print("-> Métrica: Taxa de conversão")
    print(f"   Controle (A):   {prop_result.rate_control:.2%}")
    print(f"   Tratamento (B): {prop_result.rate_treatment:.2%}")
    print(f"   Lift absoluto:  {prop_result.absolute_lift:+.2%}")
    print(f"   Lift relativo:  {prop_result.relative_lift:+.1%}")
    print(f"   IC 95% do lift: [{prop_result.ci_95[0]:+.2%}, {prop_result.ci_95[1]:+.2%}]")
    print(f"   z = {prop_result.z_stat:.3f}, p-valor = {prop_result.p_value:.4f}")
    veredito = "SIGNIFICATIVO ✅" if prop_result.significant else "não significativo ❌"
    print(f"   Resultado (alpha={alpha}): {veredito}")

    # --- Métrica 2: tempo de sessão (contínua) ---
    mean_result = welch_t_test(data.control_session_time, data.treatment_session_time, alpha=alpha)
    print("\n-> Métrica: Tempo médio de sessão (segundos)")
    print(f"   Controle (A):   {mean_result.mean_control:.1f}s")
    print(f"   Tratamento (B): {mean_result.mean_treatment:.1f}s")
    print(f"   Lift absoluto:  {mean_result.absolute_lift:+.1f}s")
    print(f"   Lift relativo:  {mean_result.relative_lift:+.1%}")
    print(f"   IC 95% do lift: [{mean_result.ci_95[0]:+.1f}s, {mean_result.ci_95[1]:+.1f}s]")
    print(f"   t = {mean_result.t_stat:.3f}, p-valor = {mean_result.p_value:.4f}")
    veredito = "SIGNIFICATIVO ✅" if mean_result.significant else "não significativo ❌"
    print(f"   Resultado (alpha={alpha}): {veredito}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Execução de exemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Passo 0: quantos usuários preciso antes de sequer rodar o teste? ---
    n_necessario = sample_size_for_proportions(
        baseline_rate=0.10,
        minimum_detectable_effect=0.02,  # quer detectar +2 p.p. (10% -> 12%)
        alpha=0.05,
        power=0.8,
    )
    print(f"Tamanho de amostra necessário por grupo: {n_necessario} usuários\n")

    # --- Passo 1: exemplo de atribuição determinística de grupo ---
    assigner = ExperimentAssigner(experiment_name="checkout_button_color", split=0.5)
    usuarios_exemplo = ["user_1", "user_2", "user_3", "user_4"]
    print("Exemplo de atribuição de grupo (hashing consistente):")
    for u in usuarios_exemplo:
        print(f"  {u} -> Grupo {assigner.assign(u).value}")
    print()

    # --- Passo 2: simula o experimento rodando com o tamanho de amostra calculado ---
    dados = simulate_experiment(
        n_per_group=n_necessario,
        control_conversion_rate=0.10,
        treatment_conversion_rate=0.12,
    )

    # --- Passo 3: analisa e imprime o relatório ---
    print_report(dados, alpha=0.05)
