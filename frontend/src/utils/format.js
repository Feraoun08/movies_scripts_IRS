export const formatRating = (value) => (value == null ? '-' : Number(value).toFixed(1));

export const formatScore = (value) => (value == null ? '-' : Math.round(Number(value)));

export const formatVotes = (value) => (value == null ? '-' : Number(value).toLocaleString('en-US'));

export const formatYear = (value) => (value == null ? '-' : value);
