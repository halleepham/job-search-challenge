"""
REQ-2 phase 2a: build the skill gazetteer (AC-2.1, AC-2.8).

Seeds from `data_jobs.job_skills` - 252 distinct pre-parsed skills, and despite
the dataset's name not data-only (26 of 30 probed non-data tech terms are
present) - then hand-extends with the areas that corpus under-covers: mobile,
the .NET ecosystem, design tooling, security tooling, and test frameworks.

Output: data/vocabulary/skills_vocabulary.json (committed). Aliases live in that
file as data, never in code (AC-2.8), so the vocabulary can be corrected without
touching the extractor.

    python -m src.build_vocabulary
"""

from __future__ import annotations

import ast
import collections
import json
from pathlib import Path

OUT = Path("data/vocabulary/skills_vocabulary.json")
SEED_DATASET = "lukebarousse/data_jobs"

#: Terms absent from or under-represented in the data-roles seed. Added by hand
#: so software, mobile, security and design roles are not silently unmatchable.
EXTENSIONS = [
    # .NET / Microsoft
    "dotnet", "asp.net", "entity framework", "blazor", "xamarin", "visual basic",
    # mobile
    "ios", "android", "react native", "flutter", "swiftui", "jetpack compose", "xcode",
    # web / frontend
    "next.js", "svelte", "redux", "webpack", "vite", "tailwind", "sass", "jquery",
    "html", "css", "web components", "responsive design",
    # backend / frameworks
    "fastapi", "flask", "express", "nestjs", "rails", "laravel", "asp", "grpc",
    "rest api", "soap", "microservices", "websockets",
    # infra / devops
    "helm", "argo cd", "circleci", "github actions", "gitlab ci", "travis ci",
    "puppet", "chef", "vagrant", "nginx", "apache", "consul", "vault", "istio",
    "prometheus", "grafana", "datadog", "new relic", "splunk", "cloudformation",
    "pulumi", "openshift", "rancher", "serverless", "lambda", "ec2", "s3", "eks",
    # security
    "penetration testing", "owasp", "siem", "soc", "nessus", "burp suite", "metasploit",
    "wireshark", "nmap", "cissp", "iso 27001", "nist", "zero trust", "iam", "okta",
    "active directory", "ldap", "oauth", "saml", "encryption", "pki", "firewall",
    "vulnerability management", "incident response", "threat modeling",
    # testing / QA
    "selenium", "cypress", "playwright", "jest", "mocha", "pytest", "junit", "testng",
    "cucumber", "appium", "postman", "soapui", "load testing", "regression testing",
    "test automation", "unit testing", "integration testing",
    # data / ml extras
    "dbt", "dagster", "prefect", "great expectations", "mlflow", "kubeflow", "sagemaker",
    "hugging face", "langchain", "vector database", "feature engineering", "nlp",
    "computer vision", "time series", "a/b testing", "experimentation",
    # design / product
    "figma", "sketch", "adobe xd", "invision", "wireframing", "prototyping",
    "user research", "accessibility", "wcag",
    # practice / process
    "agile", "scrum", "kanban", "ci/cd", "code review", "pair programming",
    "technical writing", "system design", "distributed systems", "api design",
]

#: Genuine synonyms only - never near-meanings (AC-2.8). Written canonical <- aliases.
ALIASES = {
    "kubernetes": ["k8s"],
    "react": ["react.js", "reactjs"],
    "node.js": ["node", "nodejs"],
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "python": ["py", "python3"],
    "go": ["golang"],
    "postgresql": ["postgres", "psql"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "google cloud": ["gcp", "google cloud platform"],
    "aws": ["amazon web services"],
    "azure": ["microsoft azure"],
    "tensorflow": ["tf"],
    "power bi": ["powerbi", "microsoft power bi"],
    "ci/cd": ["cicd", "ci-cd", "continuous integration"],
    "dotnet": [".net", "asp.net core", "c# .net"],
    "sql server": ["mssql", "microsoft sql server"],
    "business intelligence": ["bi"],
    "natural language processing": ["nlp"],
    "amazon redshift": ["redshift"],
    "elasticsearch": ["elastic search", "elk"],
    "objective-c": ["objective c", "objc"],
    "c#": ["csharp", "c sharp"],
    "c++": ["cpp", "cplusplus"],
    "github actions": ["gh actions"],
    "rest api": ["rest", "restful"],
    "extract transform load": ["etl", "elt"],
}


def load_seed() -> collections.Counter:
    from datasets import load_dataset

    counts: collections.Counter[str] = collections.Counter()
    for raw in load_dataset(SEED_DATASET, split="train")["job_skills"]:
        if not raw:
            continue
        try:
            for skill in ast.literal_eval(raw):
                counts[skill.strip().lower()] += 1
        except (ValueError, SyntaxError):
            continue
    return counts


def main() -> None:
    seed = load_seed()
    canonical = sorted(set(seed) | {e.lower() for e in EXTENSIONS} | set(ALIASES))

    # An alias must not also be a canonical term - that would make one surface
    # form resolve two ways depending on lookup order.
    alias_to_canon: dict[str, str] = {}
    collisions = []
    for canon, aliases in ALIASES.items():
        for alias in aliases:
            if alias in canonical and alias != canon:
                collisions.append((alias, canon))
            alias_to_canon[alias] = canon
    canonical = [c for c in canonical if c not in alias_to_canon]

    vocab = {
        "_meta": {
            "seed_dataset": SEED_DATASET,
            "seed_terms": len(seed),
            "hand_added": len([e for e in EXTENSIONS if e.lower() not in seed]),
            "canonical_terms": len(canonical),
            "alias_forms": len(alias_to_canon),
            "note": "AC-2.1/AC-2.8. Aliases are data, not code. Canonical terms and "
                    "alias forms are disjoint.",
        },
        "skills": {
            c: sorted(a for a, k in alias_to_canon.items() if k == c) for c in canonical
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(vocab, indent=1, sort_keys=True))

    print(f"seed terms from {SEED_DATASET}: {len(seed)}")
    print(f"hand-added extensions        : {vocab['_meta']['hand_added']}")
    print(f"canonical terms              : {len(canonical)}")
    print(f"alias forms                  : {len(alias_to_canon)}")
    if collisions:
        print(f"\nalias/canonical collisions resolved to canonical: {collisions}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
