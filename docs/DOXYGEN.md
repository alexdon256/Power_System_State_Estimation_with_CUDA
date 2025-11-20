# Doxygen Documentation

This project includes comprehensive API documentation generated with Doxygen.

## Building Documentation

### Prerequisites

Install Doxygen:
- **Linux (Ubuntu/Debian)**: `sudo apt-get install doxygen graphviz`
- **Linux (Fedora/RHEL)**: `sudo dnf install doxygen graphviz`
- **macOS**: `brew install doxygen graphviz`
- **Windows**: Download from https://www.doxygen.nl/download.html

**Note:** Graphviz is optional but recommended for generating class diagrams and call graphs.

### Build with CMake

```bash
# Configure with documentation enabled
cmake .. -DBUILD_DOCS=ON

# Build documentation
cmake --build . --target docs
```

The documentation will be generated in `build/docs/doxygen/html/`.

### Build Manually

```bash
# Generate documentation
doxygen Doxyfile
```

The documentation will be generated in `docs/doxygen/html/`.

## Viewing Documentation

### HTML Output

Open `docs/doxygen/html/index.html` in your web browser.

### Main Sections

- **Classes**: All classes organized by namespace
- **Files**: Source file documentation
- **Namespaces**: Namespace hierarchy
- **Modules**: Feature groups
- **Examples**: Code examples from `examples/` directory

## Documentation Structure

The Doxygen documentation includes:

1. **API Reference**
   - All public classes and functions
   - Complete parameter descriptions
   - Return value documentation
   - Usage examples

2. **Class Diagrams**
   - Inheritance hierarchies
   - Collaboration diagrams
   - Include dependencies

3. **Code Examples**
   - Examples from `examples/` directory
   - Inline code snippets

4. **Cross-References**
   - Links between related classes
   - File dependencies
   - Call graphs (if enabled)

## Adding Documentation

### Class Documentation

```cpp
/**
 * @brief State estimator for power systems
 * 
 * This class performs weighted least squares state estimation
 * using Newton-Raphson iterative solver with GPU acceleration.
 * 
 * @example basic_example.cpp
 */
class StateEstimator {
    // ...
};
```

### Function Documentation

```cpp
/**
 * @brief Estimate system state from measurements
 * 
 * Performs iterative state estimation using WLS method.
 * 
 * @param config Solver configuration (tolerance, max iterations, GPU usage)
 * @return StateEstimationResult containing estimated state and convergence info
 * 
 * @throws std::runtime_error if network or telemetry not set
 * 
 * @see estimateIncremental() for faster real-time updates
 */
StateEstimationResult estimate(const SolverConfig& config = SolverConfig());
```

### Parameter Documentation

```cpp
/**
 * @brief Set network model
 * @param network Shared pointer to network model (buses, branches, transformers)
 * @note Network must be fully configured before calling estimate()
 */
void setNetwork(std::shared_ptr<NetworkModel> network);
```

## Documentation Tags

Common Doxygen tags used in this project:

- `@brief` - Brief description
- `@param` - Parameter description
- `@return` - Return value description
- `@throws` - Exception documentation
- `@note` - Important notes
- `@warning` - Warnings
- `@see` - Cross-references
- `@example` - Code examples
- `@ingroup` - Module grouping

## Configuration

The Doxygen configuration is in `Doxyfile`. Key settings:

- **Input**: `include/`, `README.md`, `docs/API.md`
- **Output**: `docs/doxygen/html/`
- **Extract All**: Enabled (documents all public members)
- **Class Diagrams**: Enabled (requires Graphviz)
- **Source Browser**: Enabled
- **Search**: Enabled

## Integration with CI/CD

To generate documentation in CI/CD:

```yaml
# Example GitHub Actions
- name: Install Doxygen
  run: sudo apt-get install -y doxygen graphviz

- name: Generate Documentation
  run: |
    cmake -DBUILD_DOCS=ON ..
    cmake --build . --target docs

- name: Deploy Documentation
  uses: peaceiris/actions-gh-pages@v3
  with:
    publish_dir: ./docs/doxygen/html
```

## Troubleshooting

### Missing Class Diagrams

**Problem**: Class diagrams not generated.

**Solution**: Install Graphviz:
```bash
sudo apt-get install graphviz  # Linux
brew install graphviz          # macOS
```

### Documentation Not Updating

**Problem**: Changes not reflected in documentation.

**Solution**: Clean and rebuild:
```bash
rm -rf docs/doxygen/
cmake --build . --target docs
```

### Missing Examples

**Problem**: Examples not included in documentation.

**Solution**: Ensure `examples/` directory is accessible and contains `.cpp` files.

## See Also

- [API.md](API.md) - High-level API documentation
- [SETTERS_GETTERS.md](SETTERS_GETTERS.md) - Setters and getters reference
- [INDEX.md](INDEX.md) - Complete documentation index

