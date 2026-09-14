// Entry points of the rdna4-infer subcommands.
//
// src/main.hip is only a dispatcher: every subcommand lives in its own
// translation unit and is declared here, so the three parallel workstreams
// (docs/agentes-paralelos.md) never edit the same code. The dispatch line in
// main() is the whole coupling:
//
//   if (argc > 1 && std::strcmp(argv[1], "serve") == 0) return cmd_serve(argc - 1, argv + 1);
//
// `argc`/`argv` are the arguments *after* the subcommand name, exactly as
// main.hip passes them (argv[0] is the subcommand), and the return value is the
// process exit status: 0 ok, 1 usage/IO/model error, 2 no gfx1201 device,
// 3 insufficient VRAM.
#pragma once

// serve — OpenAI-compatible HTTP server (src/server/serve.hip,
// docs/servidor-openai.md). Allocates the model on the GPU, so it must be run
// under scripts/gpu-lock.sh like every other GPU command.
int cmd_serve(int argc, char **argv);
