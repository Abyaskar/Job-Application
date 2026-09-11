"""
In-memory substitute for a Motor (MongoDB async driver) collection.

Why this exists: the whole point of the repository layer is that business
logic depends on an *interface*, not on MongoDB specifically. Implementing
that interface twice (Motor-backed, memory-backed) instead of branching
inside repositories means:
  1. The service is runnable with zero infrastructure for review/demo.
  2. Repository code is never polluted with `if DEMO_MODE` checks.
  3. It doubles as a fast in-memory fixture for unit tests.

It intentionally implements only the operators this project actually uses
($eq, $in, $gte, $lte) -- it is not a general Mongo emulator.
"""
from __future__ import annotations

import itertools
import uuid
from copy import deepcopy
from typing import Any


def _match(doc: dict, query: dict) -> bool:
    for key, cond in query.items():
        val = doc.get(key)
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$eq" and val != target:
                    return False
                if op == "$in" and val not in target:
                    return False
                if op == "$gte" and not (val is not None and val >= target):
                    return False
                if op == "$lte" and not (val is not None and val <= target):
                    return False
                if op == "$ne" and val == target:
                    return False
        else:
            if val != cond:
                return False
    return True


class _Cursor:
    """Minimal async cursor supporting sort/skip/limit/to_list."""

    def __init__(self, docs: list[dict]):
        self._docs = docs

    def sort(self, field: str, direction: int = 1):
        self._docs = sorted(self._docs, key=lambda d: d.get(field), reverse=direction < 0)
        return self

    def skip(self, n: int):
        self._docs = self._docs[n:]
        return self

    def limit(self, n: int):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, length: int | None = None):
        return deepcopy(self._docs[:length] if length else self._docs)

    def __aiter__(self):
        self._iter = iter(deepcopy(self._docs))
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class InMemoryCollection:
    def __init__(self):
        self._docs: dict[str, dict] = {}
        self._counter = itertools.count(1)

    async def insert_one(self, doc: dict):
        doc = deepcopy(doc)
        doc.setdefault("_id", str(uuid.uuid4()))
        self._docs[doc["_id"]] = doc
        return type("Result", (), {"inserted_id": doc["_id"]})()

    async def find_one(self, query: dict | None = None):
        query = query or {}
        if "_id" in query and len(query) == 1:
            doc = self._docs.get(query["_id"])
            return deepcopy(doc) if doc else None
        for doc in self._docs.values():
            if _match(doc, query):
                return deepcopy(doc)
        return None

    def find(self, query: dict | None = None):
        query = query or {}
        matched = [d for d in self._docs.values() if _match(d, query)]
        return _Cursor(matched)

    async def update_one(self, query: dict, update: dict, upsert: bool = False):
        doc = await self.find_one(query)
        if not doc:
            if upsert:
                new_doc = {**query, **update.get("$set", {})}
                await self.insert_one(new_doc)
            return type("Result", (), {"modified_count": 0, "matched_count": 0})()
        set_fields = update.get("$set", {})
        inc_fields = update.get("$inc", {})
        push_fields = update.get("$push", {})
        stored = self._docs[doc["_id"]]
        stored.update(set_fields)
        for k, v in inc_fields.items():
            stored[k] = stored.get(k, 0) + v
        for k, v in push_fields.items():
            stored.setdefault(k, []).append(v)
        return type("Result", (), {"modified_count": 1, "matched_count": 1})()

    async def delete_one(self, query: dict):
        doc = await self.find_one(query)
        if doc:
            del self._docs[doc["_id"]]
            return type("Result", (), {"deleted_count": 1})()
        return type("Result", (), {"deleted_count": 0})()

    async def count_documents(self, query: dict | None = None):
        query = query or {}
        return len([d for d in self._docs.values() if _match(d, query)])

    async def create_index(self, *args, **kwargs):
        return "noop-index"


class InMemoryDatabase:
    """Mimics `motor_client[db_name]` attribute-style collection access."""

    def __init__(self):
        self._collections: dict[str, InMemoryCollection] = {}

    def __getitem__(self, name: str) -> InMemoryCollection:
        return self._collections.setdefault(name, InMemoryCollection())

    def __getattr__(self, name: str) -> InMemoryCollection:
        return self[name]
