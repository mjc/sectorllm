{
  description = "Development shell for sectorllm";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3.withPackages (ps: [ ps.numpy ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.coreutils
              pkgs.curl
              pkgs.gnumake
              pkgs.imagemagick
              pkgs.nasm
              pkgs.qemu
              pkgs.xxd
              python
            ];
          };
        }
      );
    };
}
