import { tool } from "@opencode-ai/plugin"
import path from "path"

export default tool({
  description: "Fetches the content of a webpage given a URL using the Serper API",
  args: {
    url: tool.schema.string().describe("URL to fetch"),
  },
  async execute(args, context) {
    const script = path.join(context.worktree, ".opencode/tools/web_fetch.py")
    const result = await Bun.$`python ${script} -u ${args.url}`.text()
    return result.trim()
  },
})
