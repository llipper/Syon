"""Conteúdo curriculum Master/Senior — programação, segurança e arquitetura."""

from __future__ import annotations

# Cada entrada: text (instrução completa), metadata dict
# Mapeamento source -> lista de entradas (alinhado a configs/dataset/composition.yaml)

ENTRIES: dict[str, dict[str, list[dict]]] = {
    "programming": {
        "github_repositories": [
            {
                "text": (
                    "<|system|>Você é Syon, engenheiro senior especializado em programação segura.\n"
                    "<|user|>Implemente Circuit Breaker em Python para microsserviços com fallback e métricas.\n"
                    "<|assistant|>```python\n"
                    "from dataclasses import dataclass, field\n"
                    "from enum import Enum\n"
                    "import time\n"
                    "from typing import Callable, TypeVar\n\n"
                    "T = TypeVar('T')\n\n"
                    "class State(Enum):\n"
                    "    CLOSED = 'closed'\n"
                    "    OPEN = 'open'\n"
                    "    HALF_OPEN = 'half_open'\n\n"
                    "@dataclass\n"
                    "class CircuitBreaker:\n"
                    "    failure_threshold: int = 5\n"
                    "    recovery_timeout: float = 30.0\n"
                    "    _failures: int = field(default=0, init=False)\n"
                    "    _state: State = field(default=State.CLOSED, init=False)\n"
                    "    _opened_at: float = field(default=0.0, init=False)\n\n"
                    "    def call(self, fn: Callable[[], T], fallback: Callable[[], T]) -> T:\n"
                    "        if self._state == State.OPEN:\n"
                    "            if time.monotonic() - self._opened_at >= self.recovery_timeout:\n"
                    "                self._state = State.HALF_OPEN\n"
                    "            else:\n"
                    "                return fallback()\n"
                    "        try:\n"
                    "            result = fn()\n"
                    "            self._on_success()\n"
                    "            return result\n"
                    "        except Exception:\n"
                    "            self._on_failure()\n"
                    "            return fallback()\n\n"
                    "    def _on_success(self) -> None:\n"
                    "        self._failures = 0\n"
                    "        self._state = State.CLOSED\n\n"
                    "    def _on_failure(self) -> None:\n"
                    "        self._failures += 1\n"
                    "        if self._failures >= self.failure_threshold:\n"
                    "            self._state = State.OPEN\n"
                    "            self._opened_at = time.monotonic()\n"
                    "```\n"
                    "Padrão: fail-fast, isolamento de falhas, observabilidade via contadores exportáveis."
                ),
                "metadata": {"level": "senior", "topic": "system_design_patterns", "language": "python"},
            },
            {
                "text": (
                    "<|system|>Syon — especialista Rust systems programming.\n"
                    "<|user|>Worker pool thread-safe em Rust com canal MPSC e graceful shutdown.\n"
                    "<|assistant|>Use `std::sync::mpsc` + `JoinHandle`. Propague `CancellationToken` "
                    "(tokio::sync ou atomic bool). Nunca ignore `Send` bounds. "
                    "Limite fila para backpressure. Padrão senior: RAII guard que sinaliza shutdown "
                    "no Drop do pool owner."
                ),
                "metadata": {"level": "master", "topic": "concurrency_parallelism", "language": "rust"},
            },
        ],
        "stack_overflow_qa": [
            {
                "text": (
                    "<|user|>Como evitar N+1 queries em ORM sem sacrificar legibilidade?\n"
                    "<|assistant|>Senior approach: 1) `select_related`/`prefetch_related` (Django) "
                    "ou eager loading explícito; 2) DataLoader pattern em GraphQL; "
                    "3) Projeções DTO em vez de entidades completas; 4) Métricas de query count "
                    "por request em staging. Anti-pattern: lazy load em loops."
                ),
                "metadata": {"level": "senior", "topic": "performance_optimization"},
            },
        ],
        "official_documentation": [
            {
                "text": (
                    "<|user|>Explique idempotência em APIs REST para pagamentos.\n"
                    "<|assistant|>Header `Idempotency-Key` único por operação lógica. "
                    "Servidor persiste (key, request_hash, response, status) com TTL. "
                    "Replay retorna mesma resposta sem reprocessar. "
                    "Chave deve ser UUID v4 do cliente. Master: combine com outbox pattern "
                    "para consistência eventual com gateway."
                ),
                "metadata": {"level": "master", "topic": "api_design_rest_graphql"},
            },
        ],
        "public_domain_books": [
            {
                "text": (
                    "<|user|>SOLID em código legado — por onde começar?\n"
                    "<|assistant|>1) Characterization tests; 2) Extract interface no ponto de extensão; "
                    "3) SRP: separar I/O de regra de negócio; 4) DIP: injete abstrações nos hot paths; "
                    "5) Strangler fig para módulos monolíticos. Nunca big-bang rewrite."
                ),
                "metadata": {"level": "senior", "topic": "refactoring_legacy"},
            },
        ],
        "educational_course_code": [
            {
                "text": (
                    "<|user|>Property-based testing com Hypothesis — exemplo senior.\n"
                    "<|assistant|>```python\n"
                    "from hypothesis import given, strategies as st\n\n"
                    "@given(st.lists(st.integers()))\n"
                    "def test_sort_idempotent(xs):\n"
                    "    assert sorted(sorted(xs)) == sorted(xs)\n"
                    "```\n"
                    "Combine com shrinking automático. Use `assume()` para pré-condições."
                ),
                "metadata": {"level": "senior", "topic": "testing_tdd_property", "language": "python"},
            },
        ],
    },
    "cybersecurity": {
        "cve_cwe_cvss_documents": [
            {
                "text": (
                    "<|user|>Analise CWE-502 em API Python e proponha remediação senior.\n"
                    "<|assistant|>CWE-502: Deserialização insegura. Risco CVSS alto se entrada é user-controlled. "
                    "Remediação: JSON/MessagePack com schema validation (pydantic), "
                    "nunca pickle/yaml.load sem SafeLoader. Defense in depth: WAF + input size limits + "
                    "allowlist de tipos. Auditoria: grep `pickle.loads|yaml.load` no CI."
                ),
                "metadata": {"level": "master", "cwe": "CWE-502", "topic": "sast_dast_methodology"},
            },
        ],
        "security_research_papers": [
            {
                "text": (
                    "<|user|>Resumo aplicável de Zero Trust Architecture (NIST SP 800-207).\n"
                    "<|assistant|>Princípios: 1) Nunca confie, sempre verifique; 2) Assume breach; "
                    "3) Policy Decision Point (PDP) + Policy Enforcement Point (PEP); "
                    "4) Micro-segmentação; 5) MFA contínuo; 6) Device posture. "
                    "Implementação: service mesh mTLS + SPIFFE/SPIRE + OPA para ABAC."
                ),
                "metadata": {"level": "master", "topic": "zero_trust_architecture"},
            },
        ],
        "vulnerability_reports": [
            {
                "text": (
                    "<|user|>Template de relatório de vulnerabilidade para equipe de desenvolvimento.\n"
                    "<|assistant|>Seções: Resumo executivo | CWE/CVE | CVSS 3.1 vector | "
                    "Proof of concept sanitizado | Impacto negócio | Remediação com código | "
                    "Prazo SLA por severidade | Regressão test | Referências OWASP."
                ),
                "metadata": {"level": "senior", "topic": "sast_dast_methodology"},
            },
        ],
        "owasp_top_10": [
            {
                "text": (
                    "<|user|>OWASP A01 Broken Access Control — checklist arquiteto senior.\n"
                    "<|assistant|>□ Deny by default □ RBAC/ABAC centralizado □ "
                    "IDOR test em todos os recursos □ JWT: validar aud/iss/exp □ "
                    "BOLA/BFLA em APIs □ Rate limit em endpoints sensíveis □ "
                    "Audit log imutável □ Testes de autorização horizontal/vertical automatizados."
                ),
                "metadata": {"level": "master", "topic": "owasp_asvs", "owasp": "A01:2021"},
            },
        ],
        "compliance_manuals": [
            {
                "text": (
                    "<|user|>GDPR Art. 32 — medidas técnicas para sistema SaaS.\n"
                    "<|assistant|>Criptografia em trânsito (TLS 1.3) e repouso (AES-256-GCM); "
                    "pseudonimização de PII; RBAC; logging de acesso; DPIA documentada; "
                    "right to erasure com cascade delete + backup policy; DPA com subprocessadores."
                ),
                "metadata": {"level": "senior", "topic": "compliance_gdpr_hipaa_pci"},
            },
        ],
        "sanitized_attack_logs": [
            {
                "text": (
                    "<|user|>Correlacionar IoCs em log SIEM — metodologia.\n"
                    "<|assistant|>1) Normalizar timestamps UTC; 2) Hash IOCs (IP, domain, file hash); "
                    "3) MITRE ATT&CK mapping; 4) Kill chain reconstruction; "
                    "5) False positive scoring; 6) Containment playbook por TTP."
                ),
                "metadata": {"level": "master", "topic": "incident_response_nist"},
            },
        ],
        "digital_forensics": [
            {
                "text": (
                    "<|user|>Cadeia de custódia digital — requisitos forense.\n"
                    "<|assistant|>Documentar: quem coletou, quando, ferramenta, hash SHA-256 original, "
                    "write-blocker, imagem forense verificada, armazenamento WORM, "
                    "acesso logado. Admissibilidade depende de integridade comprovada."
                ),
                "metadata": {"level": "senior", "topic": "malware_analysis_defensive"},
            },
        ],
    },
    "complementary": {
        "cryptography_mathematics": [
            {
                "text": (
                    "<|user|>Quando usar AES-GCM vs ChaCha20-Poly1305?\n"
                    "<|assistant|>AES-GCM: hardware AES-NI disponível (Intel/AMD server). "
                    "ChaCha20: mobile/embedded sem AES accel. Ambos: nonce único por mensagem, "
                    "nunca reuse key+nonce. Master: key rotation via KMS + envelope encryption."
                ),
                "metadata": {"level": "master", "topic": "cryptography_applied"},
            },
        ],
        "computer_networks": [
            {
                "text": (
                    "<|user|>CAP theorem — implicações práticas para arquiteto.\n"
                    "<|assistant|>Partição de rede força escolha CP ou AP. "
                    "CP: etcd, ZooKeeper (consistência > disponibilidade). "
                    "AP: Cassandra, Dynamo-style (disponibilidade + eventual consistency). "
                    "PACELC: mesmo sem partition, latência vs consistência."
                ),
                "metadata": {"level": "master", "topic": "cap_brewer_distributed"},
            },
        ],
        "operating_systems": [
            {
                "text": (
                    "<|user|>Linux namespaces e cgroups — base de containers.\n"
                    "<|assistant|>Namespaces: isolamento PID, net, mount, UTS, IPC, user. "
                    "cgroups v2: limites CPU/mem/io. Container runtime (containerd) compõe "
                    "OCI spec + seccomp/AppArmor. Senior: não é VM — kernel compartilhado."
                ),
                "metadata": {"level": "senior", "topic": "devops_gitops_sre"},
            },
        ],
        "software_architecture": [
            {
                "text": (
                    "<|user|>Event-driven architecture — quando usar vs REST síncrono?\n"
                    "<|assistant|>EDA: desacoplamento temporal, picos de carga, audit trail natural, "
                    "Sagas para transações distribuídas. REST: consultas síncronas simples, "
                    "baixa latência read. Master: outbox + message broker (Kafka) + idempotent consumers + "
                    "schema registry (Avro/Protobuf)."
                ),
                "metadata": {"level": "master", "topic": "microservices_event_driven"},
            },
            {
                "text": (
                    "<|user|>C4 Model — níveis de documentação de arquitetura.\n"
                    "<|assistant|>1) System Context; 2) Container; 3) Component; 4) Code (opcional). "
                    "Senior entrega Context+Container para stakeholders, Component para devs. "
                    "Complemente com ADRs (Architecture Decision Records)."
                ),
                "metadata": {"level": "senior", "topic": "software_architecture_styles"},
            },
        ],
        "devops_cloud": [
            {
                "text": (
                    "<|user|>GitOps com Kubernetes — princípios senior.\n"
                    "<|assistant|>Git como source of truth; ArgoCD/Flux reconcilia cluster; "
                    "PR review para mudanças infra; drift detection; "
                    "sealed-secrets/external-secrets para credenciais; "
                    "policy-as-code (Kyverno/OPA)."
                ),
                "metadata": {"level": "master", "topic": "devops_gitops_sre"},
            },
        ],
    },
    "training_quality": {
        "synthetic_ia_generated": [
            {
                "text": (
                    "<|user|>Code review senior: identifique problemas neste handler.\n"
                    "```python\n"
                    "@app.post('/transfer')\n"
                    "def transfer(from_id, to_id, amount):\n"
                    "    db.execute(f\"UPDATE accounts SET bal=bal-{amount} WHERE id={from_id}\")\n"
                    "    db.execute(f\"UPDATE accounts SET bal=bal+{amount} WHERE id={to_id}\")\n"
                    "```\n"
                    "<|assistant|>Críticos: 1) SQL injection (f-string); 2) Sem transação ACID; "
                    "3) Sem validação saldo; 4) Sem authn/authz; 5) Sem idempotency; 6) Float para dinheiro. "
                    "Fix: parameterized queries, `BEGIN/COMMIT`, DECIMAL, auth middleware, idempotency key."
                ),
                "metadata": {"level": "master", "topic": "code_review_senior"},
            },
        ],
        "anonymized_real_cases": [
            {
                "text": (
                    "<|user|>Post-mortem: outage 4h em API de pagamentos — estrutura.\n"
                    "<|assistant|>Timeline UTC | Impacto (SLA, clientes) | Root cause (5 whys) | "
                    "Contributing factors | What went well | Action items (owner, due) | "
                    "Blameless culture | Métricas MTTR/MTBF atualizadas."
                ),
                "metadata": {"level": "senior", "topic": "architecture_review"},
            },
        ],
        "human_specialist_validation": [
            {
                "text": (
                    "<|user|>Threat modeling STRIDE para API de upload de arquivos.\n"
                    "<|assistant|>Spoofing: auth forte | Tampering: assinatura HMAC arquivo | "
                    "Repudiation: audit log | Info Disclosure: scan malware, sem list dir | "
                    "DoS: size limit, rate limit | Elevation: sandbox, sem exec no host."
                ),
                "metadata": {"level": "master", "topic": "threat_modeling_stride"},
            },
        ],
        "security_feedback": [
            {
                "text": (
                    "<|user|>Auditoria PCI-DSS em microserviço de cartão — escopo.\n"
                    "<|assistant|>CDE boundary definition | Tokenização vs armazenamento PAN | "
                    "TLS 1.2+ everywhere | Segregação rede | ASV scans | "
                    "Log retention 12 meses | MFA admin | Key management HSM."
                ),
                "metadata": {"level": "master", "topic": "compliance_gdpr_hipaa_pci"},
            },
        ],
    },
}