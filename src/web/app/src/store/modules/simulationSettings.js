const initialState = () => ({
  simulationPrecision: "6",
  centerFrequency: "500000",
  isSteadySimulation: true,
  materialProperties: {
    water: {
      vp: "1500",
      rho: "1000",
      alpha: "0",
      renderColor: "#2e86ab",
    },
    brain: {
      vp: "1560",
      rho: "1040",
      alpha: "0.3",
      renderColor: "#db504a",
    },
    corticalBone: {
      vp: "2800",
      rho: "1850",
      alpha: "4",
      renderColor: "#faf0ca",
    },
    tumor: {
      vp: "1650",
      rho: "1150",
      alpha: "0.8",
      renderColor: "#94332f",
    },
  },
});
export default {
  namespaced: true,
  state: initialState,
  mutations: {
    reset(state) {
      Object.assign(state, initialState());
    },
    updateMaterialProperty(state, { material, prop, value }) {
      if (state.materialProperties[material]) {
        state.materialProperties[material][prop] = value;
      }
    },
    setSimulationPrecision(state, payload) {
      state.simulationPrecision = payload;
    },
    setCenterFrequency(state, payload) {
      state.centerFrequency = payload;
    },
    setIsSteadySimulation(state, payload) {
      state.isSteadySimulation = payload;
    },
    setSimulationSettings(state, payload) {
      state.simulationPrecision = payload.simulationPrecision;
      state.centerFrequency = payload.centerFrequency;
      state.isSteadySimulation = payload.isSteadySimulation;
    },
  },
  actions: {
    setSimulationSettings({ commit }, payload) {
      commit("setSimulationSettings", payload);
    },
    setSimulationPrecision({ commit }, payload) {
      commit("setSimulationPrecision", payload);
    },
    setCenterFrequency({ commit }, payload) {
      commit("setCenterFrequency", payload);
    },
    setIsSteadySimulation({ commit }, payload) {
      commit("setIsSteadySimulation", payload);
    },
  },
  getters: {
    simulationPrecision(state) {
      return state.simulationPrecision;
    },
    centerFrequency(state) {
      return state.centerFrequency;
    },
    isSteadySimulation(state) {
      return state.isSteadySimulation;
    },
    simulationSettings(state) {
      return {
        simulationPrecision: state.simulationPrecision,
        centerFrequency: state.centerFrequency,
        isSteadySimulation: state.isSteadySimulation,
        materialProperties: state.materialProperties,
      };
    },
  },
};
