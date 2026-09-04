@testsnippet AquaSetup begin
    using Aqua
    using TransportMaps

end

@testitem "Package quality checks" setup = [AquaSetup] begin
    Aqua.test_all(TransportMaps)
end
