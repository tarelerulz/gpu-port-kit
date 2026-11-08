"""Build tools/exec_prof: speech-app's exec.c with a 24M-float arena, 48M-float weights, and Vulkan timestamp queries
around every dispatch. When the env var EXEC_PROF=<file> is set, it writes one line per dispatch:
    <index> <shader> <gpu ms>
Derived from ~/speech-app/port/gpu/exec.c by exact string patches (each anchor must occur once), so it follows the source."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser('~/speech-app/port/gpu/exec.c')
OUT = os.path.join(HERE, 'exec_prof')
PATCHES = [
    ("#define ARENA_FLOATS (8u << 20)", "#define ARENA_FLOATS (24u << 20)"),
    ("#define WEIGHT_FLOATS (32u << 20)", "#define WEIGHT_FLOATS (48u << 20)"),
    ("uint32_t sd = local;", "uint32_t sd = 0xffffffffu;   /* fix: spec constant 0 is not the local size */"),
    ("int main(int argc, char **argv) {",
     "static float tperiod = 1.0f; static VkQueryPool tsqp;\nint main(int argc, char **argv) {"),
    ("    vkGetPhysicalDeviceMemoryProperties(phys, &mp);",
     "    vkGetPhysicalDeviceMemoryProperties(phys, &mp);\n"
     "    { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys, &pp); tperiod = pp.limits.timestampPeriod; }"),
    ("    // descriptors and layout (shared by all shaders)",
     "    { VkQueryPoolCreateInfo qpi = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 * nops };\n"
     "      CK(vkCreateQueryPool(dev, &qpi, NULL, &tsqp)); }\n"
     "    // descriptors and layout (shared by all shaders)"),
    ("    for (int i = 0; i < nops; i++) {\n        vkCmdBindPipeline",
     "    vkCmdResetQueryPool(cb, tsqp, 0, 2 * nops);\n    for (int i = 0; i < nops; i++) {\n        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tsqp, 2 * i);\n        vkCmdBindPipeline"),
    ("        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);",
     "        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);\n"
     "        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsqp, 2 * i + 1);"),
    ('    fprintf(stderr, "gpu submit+wait %.1f ms\\n", (nowsec() - t0) * 1e3);',
     '    fprintf(stderr, "gpu submit+wait %.1f ms\\n", (nowsec() - t0) * 1e3);\n'
     '    if (getenv("EXEC_PROF")) {\n'
     '        uint64_t *res = malloc(sizeof(uint64_t) * 2 * nops);\n'
     '        CK(vkGetQueryPoolResults(dev, tsqp, 0, 2 * nops, sizeof(uint64_t) * 2 * nops, res, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));\n'
     '        FILE *pf = fopen(getenv("EXEC_PROF"), "w");\n'
     '        for (int i = 0; i < nops; i++) fprintf(pf, "%d %s %.4f\\n", i, ops[i].name, (double)(res[2 * i + 1] - res[2 * i]) * tperiod * 1e-6);\n'
     '        fclose(pf); free(res);\n'
     '    }'),
]


def main():
    src = open(SRC).read()
    for old, new in PATCHES:
        n = src.count(old)
        if n != 1:
            sys.exit(f'anchor occurs {n} times (need 1): {old[:60]!r}')
        src = src.replace(old, new)
    c = OUT + '.c'
    open(c, 'w').write(src)
    r = subprocess.run(['gcc', '-O2', '-o', OUT, c, '-lvulkan', '-L' + os.path.expanduser('~/opt/pios-vulkan/lib')],
                       capture_output=True, text=True)
    os.remove(c)
    if r.returncode:
        sys.exit(r.stderr)
    print('built', OUT)


if __name__ == '__main__':
    main()
