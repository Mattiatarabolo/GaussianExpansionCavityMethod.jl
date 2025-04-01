"""
    TwoSpinModel

A type representing the 2-Spin model on a Random Regular Graph with bimodal interactions.

# Fields
- `K::Union{Integer, Float64}`: The number of neighbors.
- `J::Float64`: The coupling strength.
- `D::Float64`: The noise strength.
"""
struct TwoSpinModel
    K::Union{Int, Float64}
    J::Float64
    D::Float64
    """
        TwoSpinModel(K, J, D)

    Construct a 2-Spin model on a Random Regular Graph with bimodal interactions.

    # Arguments
    - `K::Union{Int, Float64}`: The number of neighbors.
    - `J::Float64`: The coupling strength.
    - `D::Float64`: The noise strength.

    # Returns
    - `TwoSpinModel`: The 2-Spin model.
    """
    function TwoSpinModel(K::Union{Int, Float64}, J::Float64, D::Float64)
        new(K, J, D)
    end
end