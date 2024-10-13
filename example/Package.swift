// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "example",
    platforms: [
        .macOS(.v12)
    ],
    dependencies: [
        .package(name: "LLamaSwift", path: "../")
    ],
    targets: [
        .executableTarget(
            name: "example",
            dependencies: [
                .product(name: "LLamaSwift", package: "LLamaSwift")
            ]
        ),
    ]
)
