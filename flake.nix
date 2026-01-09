{
  description = "Pytorch with cuda enabled";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
  };
  outputs = { self, nixpkgs }:
  
  let 
   pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
  in
  { 
    devShells."x86_64-linux".default = pkgs.mkShell {
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc
        pkgs.zlib
        "/run/opengl-driver"
      ];
        
      venvDir = ".venv";
      packages = with pkgs; [
        python313    
        python313Packages.venvShellHook
        # python313Packages.uv
	# python313Packages.ruff
	# ruff
	# pyright
	# python313Packages.python-lsp-server
        python313Packages.pip
	# python313Packages.pylsp-mypy
        python313Packages.pyqt6
      ];
    };
  };
}
