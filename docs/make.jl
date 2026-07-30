using Documenter
using Literate
using UncertainTea

# --- Literate example gallery -------------------------------------------------
# `.jl` example scripts under docs/literate/ are rendered to Markdown pages that
# EXECUTE at build time (so the posterior summaries you see are real, and the
# examples double as smoke tests). Keep the sampled models SMALL: this runs in
# CI on every push to main.
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const GENERATED_DIR = joinpath(@__DIR__, "src", "generated")

isdir(GENERATED_DIR) && rm(GENERATED_DIR; recursive=true)
mkpath(GENERATED_DIR)

for script in readdir(LITERATE_DIR; join=true)
    endswith(script, ".jl") || continue
    Literate.markdown(script, GENERATED_DIR; documenter=true)
end

# --- Build the site -----------------------------------------------------------
# checkdocs=:none: the exported surface is only partially docstringed today, so
# enforcing full coverage would fail the build. Completing docstring coverage is
# tracked as a follow-up (see the doc-currency issues #213-#217); the API page
# below auto-renders whatever IS documented via @autodocs.
makedocs(;
    sitename="UncertainTea",
    modules=[UncertainTea],
    authors="shohei81",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://shohei81.github.io/uncertaintea",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Inference Overview" => "inference.md",
        "Examples" => ["Eight Schools (Non-centered)" => "generated/eight_schools.md"],
        "API Reference" => "api.md",
        "Design Notes" => "design-notes.md",
    ],
    checkdocs=:none,
)

# --- Deploy to GitHub Pages ---------------------------------------------------
# Same-repo GITHUB_TOKEN auth (no SSH DOCUMENTER_KEY needed): the docs workflow
# grants `contents: write`, which lets deploydocs push the rendered site to the
# gh-pages branch. Runs only inside GitHub Actions; a no-op locally.
deploydocs(;
    repo="github.com/shohei81/uncertaintea.git",
    devbranch="main",
    push_preview=false,
)
