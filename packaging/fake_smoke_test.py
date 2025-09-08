# This is a fake smoke test that runs on the test-infra instances after we build
# a wheel. We cannot run a real smoke test over there, because the machines are
# too old to even install a proper ffmpeg version - and without ffmpeg,
# importing torchcodec just fails. It's OK, we run our *entire* test suite on
# those wheels anyway (on other machines).

print("Success")
