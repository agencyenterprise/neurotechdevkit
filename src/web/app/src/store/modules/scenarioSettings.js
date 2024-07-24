export default {
  namespaced: true,
  state() {
    return {
      scenarioId: null,
      ctFile: null,
      scenario: null,
      isPreBuilt: true,
      availableCTs: [],
      builtInScenarios: [],
      ctSliceAxis: "",
      ctSlicePosition: null,
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
    setCTFile(state, payload) {
      state.ctFile = payload;
    },
    setCtSliceAxis(state, payload) {
      state.ctSliceAxis = payload;
    },
    setCtSlicePosition(state, payload) {
      state.ctSlicePosition = payload;
    },
  },
  actions: {
    setBuiltInScenarios({ commit }, payload) {
      commit("setBuiltInScenarios", payload);
    },
    setAvailableCTs({ commit }, payload) {
      commit("setAvailableCTs", payload);
    },
    setCtSliceAxis({ commit }, payload) {
      commit("setCtSliceAxis", payload);
    },
    setCtSlicePosition({ commit }, payload) {
      commit("setCtSlicePosition", payload);
    },
    setCTFile({ commit }, payload) {
      commit("setCTFile", payload);
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
  },
  getters: {
    isScenarioValid(state) {
      if (state.isPreBuilt) {
        return state.scenarioId !== null;
      }
      return (
        state.ctFile !== null &&
        state.ctSliceAxis !== "" &&
        state.ctSlicePosition !== null
      );
    },
    scenarioSettingsPayload(state, getters, rootState, { is2d }) {
      const payload = {
        isPreBuilt: state.isPreBuilt,
      };

      // If the scenario is pre-built, add the scenarioId and return the payload
      if (state.isPreBuilt) {
        payload.scenarioId = state.scenarioId;
        return payload;
      }

      payload.ctFile = state.ctFile;

      // If the scenario is 2D, also include the ctSliceAxis and ctSlicePosition
      if (is2d) {
        payload.ctSliceAxis = state.ctSliceAxis;
        payload.ctSlicePosition = state.ctSlicePosition;
      }

      return payload;
    },
    scenario(state) {
      return state.scenario;
    },
    scenarioId(state) {
      return state.scenarioId;
    },
    isPreBuilt(state) {
      return state.isPreBuilt;
    },
    availableCTs(state) {
      return state.availableCTs;
    },
    ctFile(state) {
      return state.ctFile;
    },
    ctSliceAxis(state) {
      return state.ctSliceAxis;
    },
    ctSlicePosition(state) {
      return state.ctSlicePosition;
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
