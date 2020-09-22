# Conda environment management scripts

[./utils.sh](utils.sh) provides utilities that makes it easy to setup torchaudio development environment.
These utilities are compatible with Linux, macOS and Windows systems.

## Usage

Souce the [./utils.sh](utils.sh) and run the commands.

1. Install conda environment and initialize conda

    ```shell
    install_conda
    init_conda
    ```

2. Create an environment

    ```shell
    create_env <TORCHAUDIO_VERSION> <PYTHON_VERSION>
    activate_env <TORCHAUDIO_VERSION> <PYTHON_VERSION>
    ```

    where `TORCHAUDIO_VERSION` is a release version or `"master"`, and `PYTHON_VERSION` is the version of python you would like to use.

    `create_env` will create environment in top-level `envs` directory.

3. Install release version

   If you used release version for `TORCHAUDIO_VERSION` in the previous step, then you can install the released binary with the following command.

   ```shell
   install_release <PYTHON_VERSION>
   ```

4. Install build dependencies

   If you are building master torchaudio from source, the following command installs dependencies of torchaudio, except `PyTorch`. This function is designed this way so that it's easy to cache the environment directory in CI (which should not include `PyTorch`).

    ```shell
    install_build_dependencies
    ```

5. Build master

   Once the dependencies are installed, you can build torchaudio with the following command.

   ```shell
   build_master
   ```

### CUDA

To build/install CUDA compatible version, set `CUDA_VERSION` environment variable, such as `10.1`.
