import os
from typing import Optional, List, Dict, Any

import pandas as pd
import networkx as nx
import lancedb
from langchain_openai import OpenAIEmbeddings


class MSGraphRAG:
    def __init__(
        self,
        output_dir: str,
        embeddings: Optional[OpenAIEmbeddings] = None,
        verbose: bool = True,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.db_path = os.path.join(self.output_dir, "lancedb")
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        self.verbose = verbose

        ent_pq = os.path.join(self.output_dir, "entities.parquet")
        rel_pq = os.path.join(self.output_dir, "relationships.parquet")
        if not (os.path.exists(ent_pq) and os.path.exists(rel_pq)):
            raise FileNotFoundError(
                f"Missing Parquet files in {self.output_dir}. "
                f"Expected entities.parquet and relationships.parquet."
            )

        self.entities_df = pd.read_parquet(ent_pq)
        self.relationships_df = pd.read_parquet(rel_pq)
        self.G = self._build_graph()

        name_col = "name" if "name" in self.entities_df.columns else (
            "title" if "title" in self.entities_df.columns else None
        )
        self.id2name: Dict[str, str] = {
            row["id"]: (row[name_col] if name_col and pd.notna(row[name_col]) else row["id"])
            for _, row in self.entities_df.iterrows()
        }

        if not os.path.isdir(self.db_path):
            raise FileNotFoundError(
                f"LanceDB directory not found: {self.db_path}. "
                f"Did you run GraphRAG indexing first?"
            )

        self.db = lancedb.connect(self.db_path)
        self.table_names = set(self.db.table_names())
        if self.verbose:
            print("LanceDB tables detected:", self.table_names)

        if "default-entity-description" in self.table_names:
            self.tbl_text_units = "default-entity-description"
        elif "default-text_unit-text" in self.table_names:
            self.tbl_text_units = "default-text_unit-text"
        else:
            self.tbl_text_units = None

        if "default-community-full_content" in self.table_names:
            self.tbl_communities = "default-community-full_content"
        else:
            self.tbl_communities = None

        if self.verbose:
            print("Picked text_units:", self.tbl_text_units)
            print("Picked communities:", self.tbl_communities)

    def _build_graph(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        for _, r in self.entities_df.iterrows():
            d = r.to_dict()
            if "id" not in d:
                continue
            G.add_node(d["id"], **d)

        rel = self.relationships_df
        src = "source" if "source" in rel.columns else "src"
        tgt = "target" if "target" in rel.columns else "dst"

        for _, r in rel.iterrows():
            u, v = r.get(src), r.get(tgt)
            if u is None or v is None:
                continue
            desc = r.get("description", "related_to")
            G.add_edge(u, v, key=desc, **r.to_dict())

        return G

    def retrieve_text_units(
        self,
        query: str,
        k_text: int = 8,
        k_comm: int = 4
    ) -> List[Dict[str, Any]]:
        qvec = self.embeddings.embed_query(query)
        out: List[Dict[str, Any]] = []

        if self.tbl_text_units and self.tbl_text_units in self.table_names:
            t = self.db.open_table(self.tbl_text_units)
            for h in t.search(qvec).limit(k_text).to_list():
                out.append(
                    {
                        "source": "text_unit",
                        "id": h.get("id"),
                        "score": float(h.get("_distance", 0.0)),
                        "text": (
                            h.get("text")
                            or h.get("description")
                            or h.get("content")
                            or h.get("chunk")
                            or ""
                        ),
                        "meta": h,
                    }
                )

        if k_comm > 0 and self.tbl_communities and self.tbl_communities in self.table_names:
            t = self.db.open_table(self.tbl_communities)
            for h in t.search(qvec).limit(k_comm).to_list():
                out.append(
                    {
                        "source": "community",
                        "id": h.get("id"),
                        "score": float(h.get("_distance", 0.0)),
                        "text": (
                            h.get("full_content")
                            or h.get("text")
                            or h.get("summary")
                            or ""
                        ),
                        "meta": h,
                    }
                )

        out.sort(key=lambda r: r["score"])
        return out

    def get_entity_context(self, ent_id: str) -> str:
        """Returnează descriere + relațiile entității."""
        lines = []

        # detalii entitate
        ent_row = self.entities_df.loc[self.entities_df["id"] == ent_id]
        if not ent_row.empty:
            for col in ["description", "summary", "title", "name"]:
                if col in ent_row.columns and pd.notna(ent_row.iloc[0][col]):
                    lines.append(f"{self.id2name.get(ent_id, ent_id)}: {ent_row.iloc[0][col]}")
                    break

        # relații
        rel_rows = self.relationships_df[
            (self.relationships_df["source"] == ent_id) | (self.relationships_df["target"] == ent_id)
        ]

        for _, r in rel_rows.iterrows():
            src = self.id2name.get(r["source"], r["source"])
            tgt = self.id2name.get(r["target"], r["target"])
            rel_text = r["description"] if pd.notna(r["description"]) else "related_to"
            lines.append(f"{src} — {rel_text} — {tgt}")

        return "\n".join(lines)
