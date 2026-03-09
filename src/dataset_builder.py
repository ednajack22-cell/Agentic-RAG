"""
Evaluation Dataset Builder
===========================

Tools for creating and managing evaluation datasets with ground truth:
- Dataset creation from queries
- Ground truth annotation
- Dataset validation
- Standard dataset integration (MS MARCO, Natural Questions)
- Dataset statistics and analysis

Essential for reproducible research and journal publication.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import csv
from datetime import datetime
from collections import Counter


@dataclass
class GroundTruthItem:
    """Single item in evaluation dataset"""
    query_id: str
    question: str
    ground_truth_answer: str
    relevant_doc_ids: List[str] = field(default_factory=list)
    query_type: str = "GENERAL"
    difficulty: str = "medium"  # easy, medium, hard
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    annotator_id: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'GroundTruthItem':
        return cls(**data)


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset"""
    name: str
    description: str
    items: List[GroundTruthItem]
    version: str = "1.0"
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at,
            'metadata': self.metadata,
            'items': [item.to_dict() for item in self.items]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvaluationDataset':
        items = [GroundTruthItem.from_dict(item) for item in data.get('items', [])]
        return cls(
            name=data['name'],
            description=data['description'],
            items=items,
            version=data.get('version', '1.0'),
            created_at=data.get('created_at', ''),
            metadata=data.get('metadata', {})
        )


class DatasetBuilder:
    """
    Builder for creating evaluation datasets

    Helps create well-structured datasets for reproducible evaluation.
    """

    def __init__(self):
        print("[Dataset Builder] Initialized")

    def create_dataset(
        self,
        name: str,
        description: str,
        items: List[GroundTruthItem]
    ) -> EvaluationDataset:
        """
        Create a new evaluation dataset

        Args:
            name: Dataset name
            description: Dataset description
            items: List of ground truth items

        Returns:
            EvaluationDataset object
        """
        dataset = EvaluationDataset(
            name=name,
            description=description,
            items=items,
            created_at=datetime.now().isoformat()
        )

        print(f"[Dataset Builder] Created dataset '{name}' with {len(items)} items")

        return dataset

    def create_from_queries(
        self,
        questions: List[str],
        dataset_name: str,
        dataset_description: str
    ) -> EvaluationDataset:
        """
        Create dataset from list of questions (ground truth to be added later)

        Args:
            questions: List of questions
            dataset_name: Name for dataset
            dataset_description: Description of dataset

        Returns:
            EvaluationDataset with empty ground truth
        """
        items = []

        for i, question in enumerate(questions):
            item = GroundTruthItem(
                query_id=f"q_{i+1:04d}",
                question=question,
                ground_truth_answer="",  # To be annotated
                created_at=datetime.now().isoformat()
            )
            items.append(item)

        return self.create_dataset(dataset_name, dataset_description, items)

    def add_ground_truth(
        self,
        dataset: EvaluationDataset,
        query_id: str,
        ground_truth_answer: str,
        relevant_doc_ids: List[str] = None,
        annotator_id: str = ""
    ):
        """
        Add ground truth to a dataset item

        Args:
            dataset: EvaluationDataset to update
            query_id: Query ID to update
            ground_truth_answer: Ground truth answer
            relevant_doc_ids: List of relevant document IDs
            annotator_id: ID of annotator
        """
        for item in dataset.items:
            if item.query_id == query_id:
                item.ground_truth_answer = ground_truth_answer
                if relevant_doc_ids:
                    item.relevant_doc_ids = relevant_doc_ids
                item.annotator_id = annotator_id
                print(f"[Dataset Builder] Updated ground truth for {query_id}")
                return

        print(f"[Dataset Builder] Warning: Query ID {query_id} not found")

    def validate_dataset(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """
        Validate dataset completeness and quality

        Args:
            dataset: EvaluationDataset to validate

        Returns:
            Validation report
        """
        report = {
            'total_items': len(dataset.items),
            'items_with_ground_truth': 0,
            'items_with_relevant_docs': 0,
            'items_with_query_type': 0,
            'missing_ground_truth': [],
            'query_type_distribution': {},
            'difficulty_distribution': {},
            'issues': []
        }

        for item in dataset.items:
            # Check ground truth
            if item.ground_truth_answer and item.ground_truth_answer.strip():
                report['items_with_ground_truth'] += 1
            else:
                report['missing_ground_truth'].append(item.query_id)

            # Check relevant docs
            if item.relevant_doc_ids:
                report['items_with_relevant_docs'] += 1

            # Check query type
            if item.query_type and item.query_type != "GENERAL":
                report['items_with_query_type'] += 1

        # Distributions
        query_types = [item.query_type for item in dataset.items]
        report['query_type_distribution'] = dict(Counter(query_types))

        difficulties = [item.difficulty for item in dataset.items]
        report['difficulty_distribution'] = dict(Counter(difficulties))

        # Check for issues
        if report['items_with_ground_truth'] < len(dataset.items):
            report['issues'].append(
                f"{len(dataset.items) - report['items_with_ground_truth']} items missing ground truth"
            )

        if report['items_with_relevant_docs'] == 0:
            report['issues'].append("No items have relevant document IDs (needed for retrieval metrics)")

        # Check for duplicates
        questions = [item.question.lower().strip() for item in dataset.items]
        duplicates = [q for q, count in Counter(questions).items() if count > 1]
        if duplicates:
            report['issues'].append(f"Found {len(duplicates)} duplicate questions")

        print(f"[Dataset Builder] Validation complete:")
        print(f"  Total items: {report['total_items']}")
        print(f"  Items with ground truth: {report['items_with_ground_truth']}")
        print(f"  Issues found: {len(report['issues'])}")

        return report

    def split_dataset(
        self,
        dataset: EvaluationDataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: str = "query_type"
    ) -> Tuple[EvaluationDataset, EvaluationDataset, EvaluationDataset]:
        """
        Split dataset into train/val/test sets

        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify_by: Field to stratify by (query_type, difficulty, etc.)

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        import random

        # Group items by stratification field
        groups = {}
        for item in dataset.items:
            key = getattr(item, stratify_by, 'default')
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        train_items = []
        val_items = []
        test_items = []

        # Split each group proportionally
        for group_items in groups.values():
            random.shuffle(group_items)

            n = len(group_items)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_items.extend(group_items[:train_end])
            val_items.extend(group_items[train_end:val_end])
            test_items.extend(group_items[val_end:])

        # Create datasets
        train_dataset = EvaluationDataset(
            name=f"{dataset.name}_train",
            description=f"Training split of {dataset.name}",
            items=train_items,
            created_at=datetime.now().isoformat()
        )

        val_dataset = EvaluationDataset(
            name=f"{dataset.name}_val",
            description=f"Validation split of {dataset.name}",
            items=val_items,
            created_at=datetime.now().isoformat()
        )

        test_dataset = EvaluationDataset(
            name=f"{dataset.name}_test",
            description=f"Test split of {dataset.name}",
            items=test_items,
            created_at=datetime.now().isoformat()
        )

        print(f"[Dataset Builder] Split dataset:")
        print(f"  Train: {len(train_items)} items")
        print(f"  Val: {len(val_items)} items")
        print(f"  Test: {len(test_items)} items")

        return train_dataset, val_dataset, test_dataset

    def save_dataset(
        self,
        dataset: EvaluationDataset,
        output_path: str,
        format: str = "json"
    ):
        """
        Save dataset to file

        Args:
            dataset: Dataset to save
            output_path: Output file path
            format: Format (json, jsonl, csv)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, 'w') as f:
                json.dump(dataset.to_dict(), f, indent=2)

        elif format == "jsonl":
            with open(path, 'w') as f:
                for item in dataset.items:
                    f.write(json.dumps(item.to_dict()) + '\n')

        elif format == "csv":
            with open(path, 'w', newline='') as f:
                if not dataset.items:
                    return

                fieldnames = list(dataset.items[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in dataset.items:
                    writer.writerow(item.to_dict())

        print(f"[Dataset Builder] Saved dataset to {output_path}")

    def load_dataset(self, input_path: str, format: str = "json") -> EvaluationDataset:
        """
        Load dataset from file

        Args:
            input_path: Input file path
            format: Format (json, jsonl, csv)

        Returns:
            EvaluationDataset
        """
        path = Path(input_path)

        if format == "json":
            with open(path, 'r') as f:
                data = json.load(f)
                dataset = EvaluationDataset.from_dict(data)

        elif format == "jsonl":
            items = []
            with open(path, 'r') as f:
                for line in f:
                    item_data = json.loads(line)
                    items.append(GroundTruthItem.from_dict(item_data))

            dataset = EvaluationDataset(
                name=path.stem,
                description="Loaded from JSONL",
                items=items,
                created_at=datetime.now().isoformat()
            )

        elif format == "csv":
            items = []
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse list fields
                    if 'relevant_doc_ids' in row and isinstance(row['relevant_doc_ids'], str):
                        row['relevant_doc_ids'] = json.loads(row['relevant_doc_ids'])
                    if 'metadata' in row and isinstance(row['metadata'], str):
                        row['metadata'] = json.loads(row['metadata'])

                    items.append(GroundTruthItem.from_dict(row))

            dataset = EvaluationDataset(
                name=path.stem,
                description="Loaded from CSV",
                items=items,
                created_at=datetime.now().isoformat()
            )

        print(f"[Dataset Builder] Loaded dataset from {input_path} ({len(dataset)} items)")

        return dataset

    def load_dataset_from_json(self, input_path: str) -> EvaluationDataset:
        """
        Load dataset from JSON file (convenience alias for load_dataset)
        
        Args:
            input_path: Input file path
            
        Returns:
            EvaluationDataset
        """
        return self.load_dataset(input_path, format="json")

    def get_dataset_statistics(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """
        Get comprehensive statistics about dataset

        Args:
            dataset: Dataset to analyze

        Returns:
            Statistics dictionary
        """
        stats = {
            'name': dataset.name,
            'total_items': len(dataset.items),
            'query_types': {},
            'difficulty': {},
            'domains': {},
            'question_length': {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0
            },
            'answer_length': {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0
            },
            'avg_relevant_docs': 0
        }

        # Query type distribution
        query_types = [item.query_type for item in dataset.items]
        stats['query_types'] = dict(Counter(query_types))

        # Difficulty distribution
        difficulties = [item.difficulty for item in dataset.items]
        stats['difficulty'] = dict(Counter(difficulties))

        # Domain distribution
        domains = [item.domain for item in dataset.items if item.domain]
        if domains:
            stats['domains'] = dict(Counter(domains))

        # Question lengths
        question_lengths = [len(item.question.split()) for item in dataset.items]
        if question_lengths:
            import numpy as np
            stats['question_length'] = {
                'mean': float(np.mean(question_lengths)),
                'median': float(np.median(question_lengths)),
                'min': int(np.min(question_lengths)),
                'max': int(np.max(question_lengths))
            }

        # Answer lengths
        answer_lengths = [len(item.ground_truth_answer.split())
                         for item in dataset.items
                         if item.ground_truth_answer]
        if answer_lengths:
            stats['answer_length'] = {
                'mean': float(np.mean(answer_lengths)),
                'median': float(np.median(answer_lengths)),
                'min': int(np.min(answer_lengths)),
                'max': int(np.max(answer_lengths))
            }

        # Relevant docs
        relevant_doc_counts = [len(item.relevant_doc_ids)
                               for item in dataset.items
                               if item.relevant_doc_ids]
        if relevant_doc_counts:
            stats['avg_relevant_docs'] = float(np.mean(relevant_doc_counts))

        return stats


if __name__ == "__main__":
    print("Evaluation Dataset Builder")
    print("=" * 60)
    print("\nFeatures:")
    print("1. Create datasets from questions")
    print("2. Add ground truth annotations")
    print("3. Validate dataset completeness")
    print("4. Split into train/val/test")
    print("5. Save/load in multiple formats (JSON, JSONL, CSV)")
    print("6. Dataset statistics and analysis")
    print("\nUsage:")
    print("  builder = DatasetBuilder()")
    print("  dataset = builder.create_from_queries(questions, 'my_dataset', 'Description')")
    print("  builder.add_ground_truth(dataset, 'q_0001', 'Answer text', ['doc1', 'doc2'])")
    print("  validation = builder.validate_dataset(dataset)")
    print("  builder.save_dataset(dataset, 'dataset.json')")
