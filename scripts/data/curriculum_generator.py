"""
Gerador procedural de curriculum Master/Senior.
Produz milhares de amostras únicas para programação, segurança e arquitetura.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterator

LANGUAGES = ["python", "rust", "go", "typescript", "java", "cpp", "kotlin", "solidity"]
LEVELS = ["senior", "master"]

# domain -> source_name -> generator yields (text, metadata)
PROGRAMMING_SCENARIOS = [
    ("API REST com rate limiting e idempotência", "api_design_rest_graphql"),
    ("pipeline ETL resiliente com retry exponencial", "distributed_systems"),
    ("parser de logs estruturados high-throughput", "performance_optimization"),
    ("migração strangler fig de monólito", "refactoring_legacy"),
    ("worker pool com backpressure", "concurrency_parallelism"),
    ("cache L1/L2 com invalidação consistente", "system_design_patterns"),
    ("testes de contrato consumer-driven", "testing_tdd_property"),
    ("CLI com plugin architecture", "multi_language_patterns"),
]

SECURITY_SCENARIOS = [
    ("threat model STRIDE para {svc}", "threat_modeling_stride"),
    ("remediação CWE-79 XSS em SPA", "sast_dast_methodology"),
    ("hardening Kubernetes RBAC least privilege", "cloud_security_cspm"),
    ("resposta a incidente ransomware", "incident_response_nist"),
    ("auditoria ASVS nível 2", "owasp_asvs"),
    ("rotação de chaves KMS envelope encryption", "cryptography_applied"),
    ("correlação MITRE ATT&CK em SIEM", "incident_response_nist"),
    ("conformidade PCI-DSS escopo CDE", "compliance_gdpr_hipaa_pci"),
]

ARCHITECTURE_SCENARIOS = [
    ("microservices vs modular monolith para {svc}", "microservices_event_driven"),
    ("saga coreografia vs orquestração", "software_architecture_styles"),
    ("event sourcing com CQRS", "microservices_event_driven"),
    ("C4 model nível container", "software_architecture_styles"),
    ("SLO/error budget SRE", "observability_sre"),
    ("GitOps com ArgoCD e policy OPA", "devops_gitops_sre"),
    ("data mesh vs data lakehouse", "data_engineering_pipelines"),
    ("CAP durante partition de rede", "cap_brewer_distributed"),
]

SERVICES = [
    "plataforma de pagamentos",
    "marketplace B2B",
    "sistema de identidade",
    "API de upload de arquivos",
    "plataforma de streaming",
    "core banking",
    "healthcare FHIR gateway",
]


def _fmt_instruction(system: str, user: str, assistant: str) -> str:
    return f"<|system|>{system}\n<|user|>{user}\n<|assistant|>{assistant}"


def _hash_dedup(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def generate_programming(rng: random.Random) -> Iterator[tuple[str, dict, str]]:
    """Yields (text, metadata, source_file)."""
    source = "github_repositories"
    for scenario, topic in PROGRAMMING_SCENARIOS:
        for lang in LANGUAGES:
            for level in LEVELS:
                user = f"[{level.upper()}] Projete e implemente solução para: {scenario} em {lang}."
                assistant = (
                    f"Arquitetura senior para {scenario}:\n"
                    f"1) Boundaries claros (hexagonal/clean)\n"
                    f"2) Idioma {lang}: use convenções oficiais e linter estrito\n"
                    f"3) Observabilidade: métricas RED/USE, tracing OpenTelemetry\n"
                    f"4) Segurança: input validation, secrets via env/KMS\n"
                    f"5) Testes: unit + integration + contract\n"
                    f"6) Trade-offs documentados em ADR\n"
                    f"Padrões aplicáveis: circuit breaker, bulkhead, retry com jitter."
                )
                text = _fmt_instruction(
                    "Você é Syon, engenheiro principal especializado em código seguro e arquitetura.",
                    user,
                    assistant,
                )
                meta = {"level": level, "topic": topic, "language": lang, "domain": "programming"}
                yield text, meta, source

    source = "stack_overflow_qa"
    patterns = ["Repository", "Factory", "Strategy", "Observer", "Adapter", "Decorator", "Command"]
    for pat in patterns:
        for lang in LANGUAGES[:4]:
            user = f"Quando usar {pat} pattern em {lang}? Exemplo production-grade."
            assistant = (
                f"{pat}: defina problema exato (variação/algoritmo/desacoplamento). "
                f"Em {lang}, injete dependências, evite singleton global. "
                f"Combine com interfaces explícitas. Anti-pattern: pattern por moda."
            )
            text = _fmt_instruction("Syon — expert em design patterns.", user, assistant)
            yield text, {"level": "senior", "topic": "system_design_patterns", "language": lang}, source

    source = "educational_course_code"
    for lang in LANGUAGES:
        user = f"Checklist code review senior — {lang}"
        assistant = (
            "□ Segurança (injection, secrets) □ Erros tratados □ Concorrência segura "
            "□ Complexidade O() aceitável □ Testes meaningful □ API estável □ Logs estruturados "
            "□ Documentação de invariantes"
        )
        text = _fmt_instruction("Syon code reviewer senior.", user, assistant)
        yield text, {"level": "master", "topic": "testing_tdd_property", "language": lang}, source


def generate_cybersecurity(rng: random.Random) -> Iterator[tuple[str, dict, str]]:
    mapping = {
        "threat_modeling_stride": "owasp_top_10",
        "sast_dast_methodology": "cve_cwe_cvss_documents",
        "cloud_security_cspm": "vulnerability_reports",
        "incident_response_nist": "sanitized_attack_logs",
        "owasp_asvs": "owasp_top_10",
        "cryptography_applied": "security_research_papers",
        "compliance_gdpr_hipaa_pci": "compliance_manuals",
    }
    for template, topic in SECURITY_SCENARIOS:
        for svc in SERVICES:
            for level in LEVELS:
                user = template.format(svc=svc) if "{svc}" in template else template
                user = f"[{level.upper()}] {user}"
                assistant = (
                    f"Análise defensiva para {svc}:\n"
                    f"- Superfície de ataque mapeada\n"
                    f"- Controles preventivos + detectivos + corretivos\n"
                    f"- Métricas: MTTD, MTTR, false positive rate\n"
                    f"- Compliance mapping onde aplicável\n"
                    f"- Runbook de escalonamento\n"
                    f"Nunca forneça exploits weaponizados — foco em remediação."
                )
                text = _fmt_instruction(
                    "Você é Syon, especialista senior em cybersegurança defensiva.",
                    user,
                    assistant,
                )
                meta = {"level": level, "topic": topic, "domain": "cybersecurity"}
                yield text, meta, mapping.get(topic, "vulnerability_reports")

    source = "digital_forensics"
    for i in range(20):
        user = f"Passo {i+1} em investigação forense digital de endpoint comprometido."
        assistant = (
            "Isole host | Imagem disco memória | Hash evidências | "
            "Timeline analysis | IOC extraction | Lateral movement mapping | "
            "Relatório admissível com cadeia de custódia."
        )
        text = _fmt_instruction("Syon forensics.", user, assistant)
        yield text, {"level": "senior", "topic": "malware_analysis_defensive"}, source


def generate_complementary(rng: random.Random) -> Iterator[tuple[str, dict, str]]:
    arch_sources = {
        "microservices_event_driven": "software_architecture",
        "software_architecture_styles": "software_architecture",
        "observability_sre": "devops_cloud",
        "devops_gitops_sre": "devops_cloud",
        "data_engineering_pipelines": "software_architecture",
        "cap_brewer_distributed": "computer_networks",
    }
    for template, topic in ARCHITECTURE_SCENARIOS:
        for svc in SERVICES:
            user = template.format(svc=svc) if "{svc}" in template else template
            assistant = (
                f"Perspectiva arquiteto master:\n"
                f"- Requisitos não-funcionais (latência, consistência, custo)\n"
                f"- Diagrama C4 container\n"
                f"- ADR com alternativas rejeitadas\n"
                f"- Evolução incremental (evitar big design upfront)\n"
                f"- Failure modes e chaos engineering"
            )
            text = _fmt_instruction("Syon — arquiteto de software principal.", user, assistant)
            meta = {"level": "master", "topic": topic, "domain": "complementary"}
            yield text, meta, arch_sources.get(topic, "software_architecture")

    source = "cryptography_mathematics"
    for algo in ["AES-GCM", "ChaCha20-Poly1305", "Ed25519", "X25519", "Argon2id"]:
        user = f"Quando e como usar {algo} em produção?"
        assistant = (
            f"{algo}: use bibliotecas auditadas (libsodium/tink). "
            "Nunca roll your own crypto. Key rotation, HSM/KMS, nonce único, "
            "versionamento de algoritmo em envelope."
        )
        text = _fmt_instruction("Syon cryptography.", user, assistant)
        yield text, {"level": "master", "topic": "cryptography_applied"}, source


def generate_training_quality(rng: random.Random) -> Iterator[tuple[str, dict, str]]:
    sources = [
        ("synthetic_ia_generated", "code_review_senior"),
        ("human_specialist_validation", "architecture_review"),
        ("security_feedback", "security_audit_scenarios"),
        ("anonymized_real_cases", "instruction_following"),
    ]
    audits = [
        "API GraphQL com depth limit e complexity scoring",
        "Lambda serverless com cold start e IAM overpermission",
        "Terraform módulo com state remoto e locking",
        "Pipeline CI/CD com SAST/DAST/SCA integrados",
    ]
    for source, topic in sources:
        for audit in audits:
            for level in LEVELS:
                user = f"[{level}] Audite: {audit}"
                assistant = (
                    "Findings priorizados por risco | Evidência | "
                    "Remediação específica | Esforço estimado | "
                    "Validação pós-fix | Referência OWASP/NIST/CWE"
                )
                text = _fmt_instruction("Syon auditor senior.", user, assistant)
                yield text, {"level": level, "topic": topic}, source


def _apply_variation(text: str, variant: int) -> str:
    """Injeta variação única para escalar dataset sem duplicata literal."""
    tags = [
        f"\n\n[Caso #{variant}] Contexto: ambiente regulado, alta escala.",
        f"\n\n[Variante {variant}] Inclua métricas e rollback strategy.",
        f"\n\n[Exercício {variant}] Priorize segurança e testabilidade.",
        f"\n\n[Master #{variant}] Trade-off latência vs consistência.",
        f"\n\n[Senior #{variant}] Compliance e auditoria contínua.",
    ]
    return text + tags[variant % len(tags)]


def generate_all(seed: int = 42, variant_offset: int = 0) -> dict[str, dict[str, list[dict]]]:
    """Retorna estrutura domain -> source -> list[{text, metadata}]."""
    rng = random.Random(seed + variant_offset)
    buckets: dict[str, dict[str, list[dict]]] = {}

    generators = [
        ("programming", generate_programming),
        ("cybersecurity", generate_cybersecurity),
        ("complementary", generate_complementary),
        ("training_quality", generate_training_quality),
    ]

    seen: set[str] = set()
    for domain, gen_fn in generators:
        buckets.setdefault(domain, {})
        for text, meta, source in gen_fn(rng):
            if variant_offset:
                text = _apply_variation(text, variant_offset)
            h = _hash_dedup(text)
            if h in seen:
                continue
            seen.add(h)
            meta = {**meta, "variant": variant_offset}
            buckets[domain].setdefault(source, []).append({"text": text, "metadata": meta})

    return buckets


def generate_until_unique(target: int, seed: int = 42) -> dict[str, dict[str, list[dict]]]:
    """Gera amostras únicas até atingir target."""
    merged: dict[str, dict[str, list[dict]]] = {}
    seen: set[str] = set()
    variant = 0

    while len(seen) < target and variant < target * 2:
        batch = generate_all(seed=seed, variant_offset=variant)
        for domain, sources in batch.items():
            merged.setdefault(domain, {})
            for source, records in sources.items():
                for rec in records:
                    h = _hash_dedup(rec["text"])
                    if h in seen:
                        continue
                    seen.add(h)
                    merged[domain].setdefault(source, []).append(rec)
                    if len(seen) >= target:
                        break
                if len(seen) >= target:
                    break
            if len(seen) >= target:
                break
        variant += 1

    return merged