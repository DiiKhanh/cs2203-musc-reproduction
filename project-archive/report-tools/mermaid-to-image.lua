function CodeBlock(block)
  if block.classes:includes("mermaid") then
    local caption = "Pipeline tổng thể của MuSc."
    local image = pandoc.Image(caption, "courses/CS2203/assets/musc-report/musc-pipeline.png")
    return pandoc.Para({image})
  end
end

function Header(header)
  if header.level == 1 then
    return {}
  end
end
