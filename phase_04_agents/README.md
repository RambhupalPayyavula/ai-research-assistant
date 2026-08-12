
## Known issue: ragas + langchain_community
`ragas` has a bug importing `ChatVertexAI` from a since-removed `langchain_community` path
(unrelated to anything this project uses). If dependencies are reinstalled from scratch,
re-run: `bash scripts/patch_ragas.sh`
