"""
Display names for values stored in normalized form.

Skills and titles are lowercased internally so matching is exact (AC-2.2), but
"python, sql, aws" reads like a typo in a form. These helpers render the stored
value for humans without changing what is stored or compared - every selection is
mapped straight back to its canonical form.
"""

from __future__ import annotations

#: Terms whose display form cannot be derived by capitalisation rules.
_EXACT = {
    "c++": "C++", "c#": "C#", "dotnet": ".NET", "asp.net": "ASP.NET",
    "javascript": "JavaScript", "typescript": "TypeScript", "nosql": "NoSQL",
    "mysql": "MySQL", "postgresql": "PostgreSQL", "mongodb": "MongoDB",
    "sql server": "SQL Server", "power bi": "Power BI", "bigquery": "BigQuery",
    "node.js": "Node.js", "next.js": "Next.js", "react native": "React Native",
    "github actions": "GitHub Actions", "gitlab ci": "GitLab CI",
    "ci/cd": "CI/CD", "rest api": "REST API", "graphql": "GraphQL",
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "scikit-learn": "scikit-learn",
    "pyspark": "PySpark", "mlflow": "MLflow", "kubeflow": "Kubeflow",
    "dbt": "dbt", "ios": "iOS", "macos": "macOS", "jira": "Jira",
    "hugging face": "Hugging Face", "langchain": "LangChain",
    "objective-c": "Objective-C", "swiftui": "SwiftUI", "jetpack compose": "Jetpack Compose",
    "a/b testing": "A/B testing", "iso 27001": "ISO 27001", "burp suite": "Burp Suite",
    "adobe xd": "Adobe XD", "invision": "InVision", "nestjs": "NestJS",
    "vue": "Vue", "svelte": "Svelte", "tailwind": "Tailwind", "sass": "Sass",
    "jquery": "jQuery", "webpack": "webpack", "vite": "Vite",
    "great expectations": "Great Expectations", "argo cd": "Argo CD",
    "new relic": "New Relic", "datadog": "Datadog", "openshift": "OpenShift",
    "sre": "SRE", "soc": "SOC", "siem": "SIEM",
}

#: Tokens that are acronyms and stay fully capitalised inside a longer name.
_ACRONYMS = {
    "sql", "aws", "gcp", "api", "apis", "bi", "ml", "ai", "etl", "elt", "nlp",
    "css", "html", "php", "sas", "sap", "erp", "crm", "qa", "dba", "it", "ui",
    "ux", "cli", "http", "https", "tcp", "dns", "vpn", "iam", "pki", "ldap",
    "saml", "oauth", "jwt", "rest", "soap", "grpc", "cissp", "cism", "owasp",
    "nist", "wcag", "eks", "ec2", "s3", "rds", "vpc", "gpu", "cpu", "hr",
    "ii", "iii", "iv", "sre", "soc", "siem", "seo", "kpi", "sla",
}


def pretty(value: str | None) -> str:
    """Human-readable form of a normalized value. Never used for comparison."""
    if not value:
        return ""
    key = value.strip().lower()
    if key in _EXACT:
        return _EXACT[key]
    words = []
    for word in key.split():
        if word in _ACRONYMS:
            words.append(word.upper())
        elif "." in word or "/" in word or "+" in word or "#" in word:
            words.append(_EXACT.get(word, word))
        else:
            words.append(word.capitalize())
    return " ".join(words)


def display_map(values) -> dict[str, str]:
    """``{display form: canonical}`` for populating a form control."""
    return {pretty(v): v for v in values}


def to_canonical(display_values, mapping: dict[str, str]) -> set[str]:
    """
    Selections back to canonical form.

    Anything typed free-hand is lowercased and trimmed rather than rejected -
    a user wanting a skill the vocabulary lacks should not be blocked by it.
    """
    out = set()
    for value in display_values:
        out.add(mapping.get(value, value.strip().lower()))
    return {v for v in out if v}
