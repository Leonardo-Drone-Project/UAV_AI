from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class Node:
    def tick(self) -> Status:
        raise NotImplementedError


@dataclass
class Condition(Node):
    name: str
    fn: Callable[[], bool]

    def tick(self) -> Status:
        return Status.SUCCESS if self.fn() else Status.FAILURE


@dataclass
class Action(Node):
    name: str
    fn: Callable[[], Status]

    def tick(self) -> Status:
        return self.fn()


@dataclass
class Sequence(Node):
    name: str
    children: List[Node]
    _idx: int = 0

    def tick(self) -> Status:
        while self._idx < len(self.children):
            s = self.children[self._idx].tick()
            if s == Status.SUCCESS:
                self._idx += 1
                continue
            if s == Status.FAILURE:
                self._idx = 0
                return Status.FAILURE
            return Status.RUNNING  # RUNNING
        self._idx = 0
        return Status.SUCCESS


@dataclass
class Selector(Node):
    name: str
    children: List[Node]
    _idx: int = 0

    def tick(self) -> Status:
        while self._idx < len(self.children):
            s = self.children[self._idx].tick()
            if s == Status.FAILURE:
                self._idx += 1
                continue
            if s == Status.SUCCESS:
                self._idx = 0
                return Status.SUCCESS
            return Status.RUNNING  # RUNNING
        self._idx = 0
        return Status.FAILURE
