"""Knowledge Graph reasoning — typed relations with inference.

Upgrades the simple co-occurrence entity graph to a typed relation graph
with inference capabilities:

  - Typed relations: uses, owns, is_part_of, depends_on, configured_as, etc.
  - Relation extraction from text (rule-based)
  - Path-based inference: if A uses B and B depends_on C, then A indirectly depends on C
  - Neighborhood expansion: given entity E, find all related entities within N hops

Used by the recall engine to:
  1. Expand entity queries with related entities
  2. Surface indirect relationships
  3. Provide structured context for the agent
"""

from __future__ import annotations

import re
from collections import defaultdict

from mamba_memory.core.l3.store import PersistentLayer
from mamba_memory.core.types import EntityNode, EntityRelation


# ---------------------------------------------------------------------------
# Relation types (closed set)
# ---------------------------------------------------------------------------

RELATION_TYPES = {
    "uses":          "A uses B (tool/library/service)",
    "depends_on":    "A depends on B (runtime dependency)",
    "is_part_of":    "A is part of B (component/module)",
    "configured_as": "A is configured as B (setting/value)",
    "replaces":      "A replaces/supersedes B",
    "related":       "Generic relation (co-mentioned)",
    "co_mentioned":  "Appeared in the same context",
    "created_by":    "A was created/authored by B (person)",
    "deployed_on":   "A is deployed on B (infrastructure)",
    "connects_to":   "A connects to B (network/API)",
}


# ---------------------------------------------------------------------------
# Relation extraction patterns
# ---------------------------------------------------------------------------

# Pattern: "A uses/uses B" → (A, uses, B)
_RELATION_PATTERNS: list[tuple[re.Pattern, str, bool]] = [
    # (regex, relation_type, reverse)
    # "uses" family
    (re.compile(r"(\w+)\s*(?:使用|用了|采用|选择了?)\s*(\w+)", re.I), "uses", False),
    (re.compile(r"(?:use|using|adopt|chose)\s+(\w+)", re.I), "uses", False),
    # "depends_on"
    (re.compile(r"(\w+)\s*(?:依赖|需要)\s*(\w+)", re.I), "depends_on", False),
    (re.compile(r"(\w+)\s*(?:depends?\s+on|requires)\s+(\w+)", re.I), "depends_on", False),
    # "replaces"
    (re.compile(r"(?:从|from)\s*(\w+)\s*(?:迁移到|切换到|换成|migrate\s+to|switch\s+to)\s*(\w+)", re.I), "replaces", True),
    (re.compile(r"(\w+)\s*(?:替换|取代|replaces?)\s*(\w+)", re.I), "replaces", False),
    # "deployed_on"
    (re.compile(r"(\w+)\s*(?:部署在|运行在|deployed?\s+(?:on|to)|runs?\s+on)\s*(\w+)", re.I), "deployed_on", False),
    # "connects_to"
    (re.compile(r"(\w+)\s*(?:连接|connects?\s+to)\s*(\w+)", re.I), "connects_to", False),
    # "configured_as"
    (re.compile(r"(\w+)\s*(?:端口|port)\s*[:=：是为]?\s*(\d+)", re.I), "configured_as", False),
]


def extract_relations(text: str, known_entities: set[str] | None = None) -> list[EntityRelation]:
    """Extract typed relations from text.

    Args:
        text: Input text to analyze
        known_entities: Optional set of known entity names for filtering

    Returns:
        List of extracted EntityRelation objects
    """
    relations: list[EntityRelation] = []

    for pattern, rel_type, reverse in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 2:
                a, b = groups[0], groups[1]
                if reverse:
                    a, b = b, a
                # Filter: at least one entity should be "interesting" (not a stop word)
                if len(a) < 2 or len(b) < 2:
                    continue
                if known_entities and a not in known_entities and b not in known_entities:
                    continue
                relations.append(EntityRelation(
                    from_entity=a,
                    to_entity=b,
                    relation_type=rel_type,
                    weight=1.0,
                ))

    return relations


# ---------------------------------------------------------------------------
# Graph traversal & inference
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Provides inference and traversal over the entity graph in L3.

    Key capabilities:
    - Multi-hop neighborhood expansion
    - Typed relation filtering
    - Indirect relationship inference
    - Entity similarity by graph structure
    """

    def __init__(self, store: PersistentLayer) -> None:
        self._store = store

    def get_neighborhood(
        self,
        entity: str,
        max_hops: int = 2,
        relation_types: list[str] | None = None,
    ) -> dict[str, list[tuple[str, str, int]]]:
        """Get all entities within N hops, grouped by entity name.

        Returns:
            dict mapping entity_name → [(relation_type, via_entity, hop_distance)]
        """
        visited: set[str] = set()
        result: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        frontier = [(entity, "", 0)]

        while frontier:
            current, via, depth = frontier.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)

            if depth > 0:
                result[current].append(("", via, depth))

            # Get outgoing relations
            related = self._store.get_related_entities(current, limit=50)
            for related_entity, rel_type in related:
                if related_entity.name in visited:
                    continue
                if relation_types and rel_type not in relation_types:
                    continue
                result[related_entity.name].append((rel_type, current, depth + 1))
                frontier.append((related_entity.name, current, depth + 1))

        return dict(result)

    def find_path(
        self,
        from_entity: str,
        to_entity: str,
        max_depth: int = 4,
    ) -> list[tuple[str, str, str]] | None:
        """Find a path between two entities (BFS).

        Returns list of (from, relation, to) triples, or None if no path.
        """
        if from_entity == to_entity:
            return []

        visited: set[str] = set()
        queue: list[tuple[str, list[tuple[str, str, str]]]] = [(from_entity, [])]

        while queue:
            current, path = queue.pop(0)
            if current in visited or len(path) >= max_depth:
                continue
            visited.add(current)

            related = self._store.get_related_entities(current, limit=50)
            for related_entity, rel_type in related:
                new_path = path + [(current, rel_type, related_entity.name)]
                if related_entity.name == to_entity:
                    return new_path
                if related_entity.name not in visited:
                    queue.append((related_entity.name, new_path))

        return None

    def infer_indirect_relations(
        self,
        entity: str,
        max_hops: int = 2,
    ) -> list[tuple[str, str, list[str]]]:
        """Infer indirect relationships.

        For example: if PostgreSQL uses port 5432, and App uses PostgreSQL,
        then App indirectly connects_to port 5432.

        Returns: [(related_entity, inferred_relation, path)]
        """
        neighborhood = self.get_neighborhood(entity, max_hops=max_hops)
        inferences: list[tuple[str, str, list[str]]] = []

        for target, connections in neighborhood.items():
            for rel_type, via, depth in connections:
                if depth >= 2:
                    # This is an indirect relationship
                    path_desc = [entity, via, target] if via else [entity, target]
                    inferred_rel = f"indirect_{rel_type}" if rel_type else "indirect_related"
                    inferences.append((target, inferred_rel, path_desc))

        return inferences

    def get_entity_context(self, entity: str, max_context: int = 5) -> str:
        """Build a natural language context string for an entity.

        Used to inject entity knowledge into agent prompts.
        """
        node = self._store.get_entity(entity)
        if not node:
            return ""

        parts = [f"{entity}"]
        if node.description:
            parts.append(f"({node.description})")

        # Direct relations
        related = self._store.get_related_entities(entity, limit=max_context)
        if related:
            rel_parts = []
            for rel_entity, rel_type in related:
                if rel_type == "co_mentioned":
                    continue  # Skip generic co-mentions
                rel_parts.append(f"{rel_type} {rel_entity.name}")
            if rel_parts:
                parts.append("→ " + ", ".join(rel_parts))

        return " ".join(parts)
