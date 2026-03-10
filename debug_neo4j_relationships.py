#!/usr/bin/env python3
"""Debug script to check Neo4j relationships."""

import asyncio
from neo4j import AsyncGraphDatabase

async def check_neo4j():
    """Check what's actually in Neo4j."""

    # Import settings
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from src.openrag.config import Settings

    settings = Settings()

    uri = settings.neo4j_uri
    username = settings.neo4j_username
    password = settings.neo4j_password
    database = settings.neo4j_database

    driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

    try:
        async with driver.session(database=database) as session:
            # Check entities
            print("=" * 80)
            print("ENTITIES in Neo4j:")
            print("=" * 80)
            result = await session.run("""
                MATCH (e:Entity)
                WHERE e.document_id = 'test_graph_doc'
                RETURN e.name as name, e.type as type
                ORDER BY e.type, e.name
            """)
            entities = await result.data()

            for entity in entities:
                print(f"  - {entity['name']} ({entity['type']})")

            print(f"\nTotal entities: {len(entities)}")

            # Check relationships
            print("\n" + "=" * 80)
            print("RELATIONSHIPS in Neo4j:")
            print("=" * 80)
            result = await session.run("""
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.document_id = 'test_graph_doc'
                RETURN e1.name as source, type(r) as rel_type, e2.name as target
                ORDER BY e1.name
            """)
            relationships = await result.data()

            if relationships:
                for rel in relationships:
                    print(f"  - {rel['source']} [{rel['rel_type']}] {rel['target']}")
            else:
                print("  No relationships found!")

            print(f"\nTotal relationships: {len(relationships)}")

            # Check chunks
            print("\n" + "=" * 80)
            print("CHUNKS in Neo4j:")
            print("=" * 80)
            result = await session.run("""
                MATCH (c:Chunk)
                WHERE c.document_id = 'test_graph_doc'
                RETURN c.chunk_id as chunk_id, c.content as content
            """)
            chunks = await result.data()

            for chunk in chunks:
                print(f"  - Chunk: {chunk['chunk_id']}")
                print(f"    Content: {chunk['content'][:100]}...")

            print(f"\nTotal chunks: {len(chunks)}")

            # Check MENTIONED_IN relationships
            print("\n" + "=" * 80)
            print("ENTITY -> CHUNK relationships:")
            print("=" * 80)
            result = await session.run("""
                MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
                WHERE e.document_id = 'test_graph_doc'
                RETURN e.name as entity, c.chunk_id as chunk
            """)
            mentions = await result.data()

            print(f"Total MENTIONED_IN relationships: {len(mentions)}")

    finally:
        await driver.close()

if __name__ == "__main__":
    asyncio.run(check_neo4j())
