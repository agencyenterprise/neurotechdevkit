export default {
  namespaced: true,
  state() {
    return {
      scenarioId: null,
      ctScan: null,
      scenario: null,
      isPreBuilt: true,
      availableCTs: [],
      builtInScenarios: [],
    };
  },
  mutations: {
    setAvailableCTs(state, payload) {
      state.availableCTs = payload;
    },
    setBuiltInScenarios(state, payload) {
      state.builtInScenarios = payload;
    },
    setScenario(state, payload) {
      state.scenario = payload;
    },
    setIsPreBuilt(state, payload) {
      state.isPreBuilt = payload;
    },
    setScenarioId(state, payload) {
      state.scenarioId = payload;
    },
    setCTScan(state, payload) {
      state.ctScan = payload;
    },
  },
  actions: {
    setBuiltInScenarios({ commit }, payload) {
      commit("setBuiltInScenarios", payload);
    },
    setAvailableCTs({ commit }, payload) {
      commit("setAvailableCTs", payload);
    },
    setScenario({ dispatch, commit, getters }, payload) {
      // payload will have the scenario name, we need to iterate over the
      // builtInScenarios to find the scenario and then set the scenario in the state
      const scenarios = getters.builtInScenarios;
      const scenario = scenarios[payload];
      commit("setScenarioId", scenario.scenarioSettings.scenarioId);
      commit("setScenario", scenario);
      dispatch("displaySettings/setDisplaySettings", scenario.displaySettings, {
        root: true,
      });
      dispatch("transducersSettings/setTransducers", scenario.transducers, {
        root: true,
      });
      dispatch(
        "simulationSettings/setSimulationSettings",
        scenario.simulationSettings,
        {
          root: true,
        }
      );
      dispatch("targetSettings/setTarget", scenario.target, {
        root: true,
      });
    },
    setIsPreBuilt({ commit }, payload) {
      commit("setIsPreBuilt", payload);
    },
    setCTScan({ commit, getters }, payload) {
      // payload will be the ct scan filename, we need to iterate over the
      // availableCTs to find the ct scan and then set the ct scan in the state
      const ctScans = getters.ctScans;
      const ctScan = ctScans.filter((ct) => ct.filename === payload)[0];
      commit("setCTScan", ctScan);
    },
  },
  getters: {
    scenario(state) {
      return state.scenario;
    },
    scenarioId(state) {
      return state.scenarioId;
    },
    isPreBuilt(state) {
      return state.isPreBuilt;
    },
    ctScans(state) {
      return state.availableCTs;
    },
    builtInScenarios(state) {
      return state.builtInScenarios;
    },
    builtInScenarios2d(state) {
      const scenarios = state.builtInScenarios;
      const filteredScenarios = Object.keys(scenarios)
        .filter((key) => scenarios[key].is2d)
        .reduce((obj, key) => {
          obj[key] = scenarios[key];
          return obj;
        }, {});
      return filteredScenarios;
    },
    builtInScenarios3d(state) {
      const scenarios = state.builtInScenarios;
      const filteredScenarios = Object.keys(scenarios)
        .filter((key) => !scenarios[key].is2d)
        .reduce((obj, key) => {
          obj[key] = scenarios[key];
          return obj;
        }, {});
      return filteredScenarios;
    },
  },
};
