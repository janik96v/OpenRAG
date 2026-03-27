# Neo4j Queries for Graph RAG Visualization

Open Neo4j Browser at: http://localhost:7474

## 🔍 View the Complete Graph (Recommended)

**This query shows entities AND their relationships:**

```cypher
MATCH (e1:Entity)-[r]->(e2:Entity)
WHERE e1.document_id = 'test_graph_doc'
RETURN e1, r, e2
```

**Expected Result:** 4 relationships with connected entities

## 🌐 View Full Graph with Chunks

**Shows entities, relationships, AND how entities connect to chunks:**

```cypher
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE e.document_id = 'test_graph_doc'
WITH e, c
OPTIONAL MATCH (e)-[r]->(e2:Entity)
WHERE e2.document_id = 'test_graph_doc'
RETURN e, c, r, e2
```

## 📊 Specific Relationship Queries

### Who Collaborates With Whom?
```cypher
MATCH (p1:Entity {type: 'PERSON'})-[r:COLLABORATES_WITH]->(p2:Entity {type: 'PERSON'})
WHERE p1.document_id = 'test_graph_doc'
RETURN p1.name as Person1, p2.name as Person2
```

**Result:** Dr. Sarah Chen collaborates with Dr. James Rodriguez

### Which Organizations Are Where?
```cypher
MATCH (org:Entity {type: 'ORGANIZATION'})-[r:LOCATED_IN]->(loc:Entity {type: 'LOCATION'})
WHERE org.document_id = 'test_graph_doc'
RETURN org.name as Organization, loc.name as Location
```

**Results:**
- MIT AI Lab → Cambridge Massachusetts
- Stanford University → San Francisco

### Who Works At Which Organization?
```cypher
MATCH (person:Entity {type: 'PERSON'})-[r:WORKS_AT]->(org:Entity {type: 'ORGANIZATION'})
WHERE person.document_id = 'test_graph_doc'
RETURN person.name as Person, org.name as Organization
```

### Partnership Networks
```cypher
MATCH (org1:Entity)-[r:PARTNERS_WITH]->(org2:Entity)
WHERE org1.document_id = 'test_graph_doc'
RETURN org1.name as Partner1, org2.name as Partner2
```

**Result:** Acme Corporation partners with MIT AI Lab

## 📈 Statistics

### Count All Relationship Types
```cypher
MATCH (e1:Entity)-[r]->(e2:Entity)
WHERE e1.document_id = 'test_graph_doc'
RETURN type(r) as RelationshipType, count(r) as Count
ORDER BY Count DESC
```

### Entity Distribution by Type
```cypher
MATCH (e:Entity)
WHERE e.document_id = 'test_graph_doc'
RETURN e.type as EntityType, count(e) as Count
ORDER BY Count DESC
```

## 🎨 Visualization Tips in Neo4j Browser

1. **Adjust Layout**: Drag nodes to rearrange
2. **Expand Connections**: Double-click a node to show connected nodes
3. **Color Coding**: Different entity types automatically get different colors
4. **View Properties**: Click on any node or relationship to see all properties

## 🧹 Clean Up Test Data

**When you're done exploring:**

```cypher
MATCH (n)
WHERE n.document_id = 'test_graph_doc'
DETACH DELETE n
```

This will remove all test nodes and relationships.

---

## 🔧 Troubleshooting

If you don't see relationships:

1. **Make sure you're using the correct query** - Use the first query above
2. **Check the data exists**: Run the statistics queries
3. **Verify document_id**: Ensure `document_id = 'test_graph_doc'`
4. **Clear browser cache**: Refresh Neo4j Browser

## Current Graph Summary

Based on the latest run:

- **15 Entities**:
  - 2 PERSON
  - 3 ORGANIZATION
  - 3 LOCATION
  - 7 CONCEPT

- **4 Entity-Entity Relationships**:
  - COLLABORATES_WITH (1)
  - LOCATED_IN (2)
  - PARTNERS_WITH (1)

- **15 MENTIONED_IN Relationships** (Entity → Chunk)
