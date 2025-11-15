from dataclasses import dataclass

import torch
from rdflib import Graph, URIRef, RDFS
from collections import deque

from transformers import BertTokenizer

from .ontology import find_synonyms_for, find_uri_for


@dataclass
class Token:
    """A Token represents a single token from the BERT tokenizer."""

    value: str
    """The string value of this token"""

    soft_position: int
    hard_position: int

    def __repr__(self):
        return f"{self.value}({self.soft_position}, {self.hard_position})"


class Node:
    """Represents a node of a SentenceTree."""

    def __init__(self, tokens: list[Token], parent: 'Node' = None, uri: URIRef = None, is_soft_edge=False,
                 is_target=False):
        self.__tokens = tokens
        self.__is_soft_edge = is_soft_edge
        self.__is_target = parent.is_target() if parent is not None else is_target
        self.uri = uri
        # use deque because it uses a linked list
        self.__children: deque[Node] = deque()

        if len(tokens) == 0:
            raise ValueError("A node cannot have no values")

        self.__parent = parent
        if self.__parent is not None:
            self.__parent.append(self)

    def is_target(self):
        return self.__is_target

    def get_n_hops(self):
        if self.__parent is None:
            return -1

        parent_hops = self.__parent.get_n_hops()
        n_hops = 0 if parent_hops == -1 else parent_hops

        if not self.__is_soft_edge:
            n_hops += 1

        return n_hops

    def is_soft_edge(self):
        return self.__is_soft_edge

    def get_tokens(self):
        return self.__tokens

    def last_token(self):
        return self.__tokens[-1]

    def first_token(self):
        return self.__tokens[0]

    def get_parent(self):
        return self.__parent

    def get_children(self):
        return self.__children

    def append(self, child: 'Node'):
        self.__children.append(child)

    def __subtree_str(self, node: 'Node', child_prefix='', prefix=''):
        if node.uri is not None:
            prefix += f"#{node.uri.fragment}"
        result = f"{prefix}{repr(node.__tokens)}"
        extra_info: list[str] = []

        if node.is_target():
            extra_info.append('target')
        if node.is_soft_edge():
            extra_info.append('soft')
        extra_info.append(f'hops {node.get_n_hops()}')

        if len(extra_info) > 0:
            result += f" ({', '.join(extra_info)})"

        for i in range(len(node.__children)):
            child = node.__children[i]

            is_last = i == len(node.__children) - 1
            prefix = ' └── ' if is_last else ' ├── '
            new_child_sep = ' │   '
            new_child_prefix = ' ' * len(new_child_sep) if is_last else new_child_sep

            result += f'\n{child_prefix}' + self.__subtree_str(child, child_prefix + new_child_prefix, prefix)

        return result

    def __repr__(self):
        """Traverses the tree to create a hierarchical representation of this (sub)tree."""
        return self.__subtree_str(self)


@dataclass
class SentenceTreeEmbedding:
    """For terminology, refer to https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel.forward"""
    tokens: list[str]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    token_type_ids: torch.Tensor
    target_start: int
    target_end: int
    hops: torch.Tensor
    vm: torch.Tensor


class SentenceTree:
    """A SentenceTree can be used to insert knowledge from an ontology into a sentence. A SentenceTree creates a node
    for each word (token) in a sentence, it inserts additional information from the ontology into the tree."""

    def __init__(self, sentence: str, target_start: int, target_end: int, ontology: Graph, tokenizer: BertTokenizer,
                 device: torch.device | str | int | None, hops=0, include_subclasses=True, include_superclasses=False):
        self.ontology = ontology
        self.tokenizer = tokenizer
        self.device = device
        self.__include_subclasses = include_subclasses
        self.__include_superclasses = include_superclasses
        self.__hops = hops
        self.__nodes = deque[Node]()
        self.__size = 0

        i = 0
        # append left context
        for token in self.__merge_word_parts(tokenizer.tokenize(sentence[0:target_start])):
            self.__append_root_node(token, i)
            i += 1
        # append target
        for token in self.__merge_word_parts(tokenizer.tokenize(sentence[target_start:target_end])):
            self.__append_root_node(token, i, True)
            i += 1
        # append right context
        for token in self.__merge_word_parts(tokenizer.tokenize(sentence[target_end:])):
            self.__append_root_node(token, i)
            i += 1

    def __merge_word_parts(self, tokens: list[str]):
        result: deque[list[str]] = deque()

        for token in tokens:
            if not token.startswith("##"):
                result.append([token])
            else:
                result[-1].append(token)

        return result

    def __append_root_node(self, tokens: list[str], soft_position: int, is_target=False):
        lex = self.tokenizer.convert_tokens_to_string(tokens)
        uri = find_uri_for(lex, self.ontology)
        token_objs = []

        for token in tokens:
            token_objs.append(Token(
                value=token,
                hard_position=self.__size,
                soft_position=soft_position))
            self.__size += 1

        node = Node(tokens=token_objs, uri=uri, is_target=is_target)
        self.__nodes.append(node)

        if uri is None or not is_target:
            return

        synonyms = find_synonyms_for(uri, self.ontology)
        self.__append_synonyms(node, synonyms, lex, uri)
        self.__construct_subtree(node)

    def __append_synonyms(self, node: Node, lex_synonyms: list[str], exclude_lex: str = None, uri: URIRef = None):
        for lex in lex_synonyms:
            if exclude_lex is not None and lex == exclude_lex:
                continue
            self.__append_node(lex, node, uri, is_soft_edge=True)

    def __append_node(self, lex: str, parent: Node, uri: URIRef = None, is_soft_edge=False):
        """A lexical representation from the ontology may contain spaces, the value is tokenized and a new Node is
        created that contains the tokens."""
        base_soft_pos = parent.first_token().soft_position if is_soft_edge else parent.last_token().soft_position + 1
        tokens: list[Token] = []

        for i, token in enumerate(list(filter(lambda token: token != '', self.tokenizer.tokenize(lex)))):
            tokens.append(Token(
                value=token,
                hard_position=self.__size,
                soft_position=base_soft_pos + i))
            self.__size += 1

        return Node(tokens, parent, uri=uri, is_soft_edge=is_soft_edge)

    def __construct_subtree(self, node: Node, current_hop=0):
        """Create subtree for a word of the original text"""
        uri = node.uri

        if current_hop >= self.__hops or uri is None or not isinstance(uri, URIRef):
            return

        # iterate subclasses
        if self.__include_subclasses:
            for target_uri, _, _ in self.ontology.triples((None, RDFS.subClassOf, uri)):
                if target_uri is None or not isinstance(target_uri, URIRef) or (
                        node.get_parent() is not None and node.get_parent().uri == target_uri):
                    continue

                synonyms = find_synonyms_for(target_uri, self.ontology)

                if len(synonyms) == 0:
                    continue

                # append synonyms, and recursively call this function on last synonym
                lex = synonyms[-1]
                new_node = self.__append_node(lex, node, target_uri)
                self.__append_synonyms(new_node, synonyms, lex, target_uri)
                self.__construct_subtree(new_node, current_hop + 1)

        # iterate superclasses
        if self.__include_superclasses:
            for _, _, target_uri in self.ontology.triples((uri, RDFS.subClassOf, None)):
                if target_uri is None or not isinstance(target_uri, URIRef) or (
                        node.get_parent() is not None and node.get_parent().uri == target_uri):
                    continue

                synonyms = find_synonyms_for(target_uri, self.ontology)

                if len(synonyms) == 0:
                    continue

                # append synonyms, and recursively call this function on last synonym
                lex = synonyms[-1]
                new_node = self.__append_node(lex, node, target_uri)
                self.__append_synonyms(new_node, synonyms, lex, target_uri)
                self.__construct_subtree(new_node, current_hop + 1)

    def build_embedding(self) -> SentenceTreeEmbedding:
        """Build this sentence tree into an input representation for the BERT model"""
        tokens: list[str] = []
        soft_positions: list[int] = []
        hops: list[int] = []

        def add_and_traverse(node: Node):
            for token in node.get_tokens():
                tokens.append(token.value)
                soft_positions.append(token.soft_position)
                hops.append(node.get_n_hops())

            for child in node.get_children():
                add_and_traverse(child)

        for node in self.__nodes:
            add_and_traverse(node)

        vm, target_start, target_end = self.__generate_vm_and_target_pos()

        return SentenceTreeEmbedding(
            tokens=tokens,
            input_ids=torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)], device=self.device),
            position_ids=torch.tensor([soft_positions], device=self.device),
            token_type_ids=torch.tensor([[0 for _ in tokens]], device=self.device),
            target_start=target_start,
            target_end=target_end,
            hops=torch.tensor(hops),
            vm=vm)

    def __generate_vm_and_target_pos(self) -> tuple[torch.Tensor, int, int]:
        # initialize n-by-n matrix filled with -inf, note that this matrix will be symmetric
        vm = torch.zeros(self.__size, self.__size) - torch.inf
        target_start = self.__size - 1
        target_end = 0

        # every node can see itself
        for i in range(self.__size):
            vm[i][i] = 0

        def check_target_index(i: int):
            nonlocal target_start, target_end
            if i < target_start:
                target_start = i
            if i + 1 > target_end:
                target_end = i + 1

        def set_child_visibility(node: Node):
            for value in node.get_tokens():
                i = value.hard_position
                current_parent = node.get_parent()

                if node.is_target():
                    check_target_index(i)

                while current_parent is not None:
                    for parent_value in current_parent.get_tokens():
                        j = parent_value.hard_position
                        vm[i][j] = 0
                        vm[j][i] = 0

                    if node.is_soft_edge() or current_parent.is_soft_edge():
                        break

                    current_parent = current_parent.get_parent()

            for child in node.get_children():
                set_child_visibility(child)

        for root in self.__nodes:
            for value in root.get_tokens():
                i = value.hard_position

                if root.is_target():
                    check_target_index(i)

                # words in the root sentence can see each other
                for other_root in self.__nodes:
                    for other_value in other_root.get_tokens():
                        j = other_value.hard_position
                        vm[i][j] = 0
                        vm[j][i] = 0

            for child in root.get_children():
                set_child_visibility(child)

        if target_start > target_end:
            raise ValueError("Could not locate the target after inserting knowledge")

        return vm, target_start, target_end

    def __len__(self):
        return self.__size

    def __repr__(self):
        """Creates a hierarchical string representation of the tree"""
        return '\n'.join([repr(node) for node in self.__nodes])
