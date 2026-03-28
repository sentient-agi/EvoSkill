import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description: "Returns a list of top google results using the Serper API. Use this to search the web for information before fetching specific pages.",
  args: {
    search_query: tool.schema.string().describe("Search Query"),
  },
  async execute(args, context) {
    const script = path.join(context.worktree, ".opencode/tools/web_search.py")
    const result = await Bun.$`python ${script} -q ${args.search_query}`.text()
    return result.trim()
  },
})